import os
import argparse

# Parse only the two parameters that change between runs
parser = argparse.ArgumentParser(description='Evaluate LLM')
parser.add_argument('--device', type=str, required=True, help='CUDA device number (e.g., "0" or "1")')
parser.add_argument('--model', type=str, required=True, help='Model name in format repo_name/model_name')
parser.add_argument('--curr_task_dir', type=str, required=True, help='Current task dir')
args = parser.parse_args()



# -----------------------
# GPU Settings
# -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import polars as pl
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
import torch
import torch.nn.functional as F
import warnings
import pandas as pd
import numpy as np
import pickle
import time
import random
import torch
from typing import Tuple

# -----------------------
# Warning Filters
# -----------------------

warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`. However, `temperature` is set to"
)
warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`. However, `top_p` is set to"
)
warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`. However, `top_k` is set to"
)


# -----------------------
# Global Configuration Constants
# -----------------------
HF_TOKEN = "<>"
GLOBAL_SEED = 42

P_FOR_YES_NO = 80  # Number of stay_id to sample for just Yes/No prediciton
MAX_NEW_TOKENS_FOR_YES_NO = 1
SYSTEM_PROMPT_FOR_YES_NO = "Answer only in Yes or No."

YES_PERCENTAGE = 0.5

MODEL_NAME = args.model
print(f"THE MODEL NAME THAT WAS PASSED IS: {MODEL_NAME}")
LLM_QUANTIZE = False # True: Will use a quantised LLM, False: will use the full precision.
QUANTIZE_8_BIT = False
QUANTIZE_4_BIT = False
TRUST_REMOTE_CODE = True     ####### BE VEEEEEEERY CAREFUL WITH THIS #########

STORE_ALL_LOGITS_IN_FINAL_RESULT = False

ALL_RESULTS_DIRECTORY="/home/gokul/Hier-Legal-Graph/mimic_dataset/mimiciv_dataset/2.2/Med_LLM_Fairness_3/All_Results/ED_Triage_v2/1_Prompt_Pruning_Exps"
print(f"The current ALL_RESULTS_DIRECTORY points to {ALL_RESULTS_DIRECTORY}")

CURRENT_TASK_DIRECTORY = args.curr_task_dir
print(f"THE TASK DIRECTORY NAME THAT WAS PASSED IS: {CURRENT_TASK_DIRECTORY}")

## Add a description AND also add the prompt template / function that makes the prompt
CURRENT_TASK_DESCRIPTION = "S"
print(f"THE TASK DESCRIPTION THAT WAS PASSED IS: {CURRENT_TASK_DESCRIPTION}")

demographic_dict= {'gender': ['Female', 'Male', 'Intersex'], 'race': ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN']}

# Construct the full path to the task directory (and check it exists)
CURRENT_TASK_DIRECTORY = os.path.join(ALL_RESULTS_DIRECTORY, CURRENT_TASK_DIRECTORY)

# Construct the final data path automatically
final_data_path = os.path.join(CURRENT_TASK_DIRECTORY, "ALL_DATASETS", "final_test_dataset.csv")

CURRENT_LLM_DIRECTORY = MODEL_NAME.split("/")[1]
CURRENT_LLM_DIRECTORY = os.path.join("Evaluation_Results", CURRENT_LLM_DIRECTORY)
CURRENT_LLM_DIRECTORY = os.path.join(CURRENT_TASK_DIRECTORY, CURRENT_LLM_DIRECTORY)

# If quantization is enabled but no bit-width is specified, default to 8-bit quantization.
if LLM_QUANTIZE and not (QUANTIZE_8_BIT or QUANTIZE_4_BIT):
    QUANTIZE_8_BIT = True


# -----------------------
# Assertions and Checks
# -----------------------


# Ensure that both bit options aren't enabled at the same time.
assert not (QUANTIZE_8_BIT and QUANTIZE_4_BIT), "Both 8-bit and 4-bit quantization cannot be enabled simultaneously."

assert MODEL_NAME, "The model name has to be defined."
# Model name must be in repo_name/model_name format
assert isinstance(MODEL_NAME, str) and "/" in MODEL_NAME and len(MODEL_NAME.split("/")) == 2, ("MODEL_NAME must be a non-empty string in the format 'repo_name/model_name'.")


# Validated that sampling params are positive ints
assert isinstance(P_FOR_YES_NO, int) and P_FOR_YES_NO > 0, "P_FOR_YES_NO must be a positive integer."

# assert isinstance(P_FOR_EXPLANATIONS, int) and P_FOR_EXPLANATIONS > 0, "P_FOR_EXPLANATIONS must be a positive integer."

# Validate that YES_PERCENTAGE is between [0, 1]
assert isinstance(YES_PERCENTAGE, (int, float)) and 0 <= YES_PERCENTAGE <= 1, "YES_PERCENTAGE must be between 0 and 1."
# Validate MAX_NEW_TOKENS_FOR_YES_NO is exactly 1
assert MAX_NEW_TOKENS_FOR_YES_NO == 1, "MAX_NEW_TOKENS_FOR_YES_NO must be exactly 1."

assert CURRENT_TASK_DIRECTORY, "The current task directory has to be defined."
assert CURRENT_TASK_DESCRIPTION.lower().strip() != """""", "Current task description should not be blank."

# Ensure ALL_RESULTS_DIRECTORY exists
assert os.path.exists(ALL_RESULTS_DIRECTORY), f"ALL_RESULTS_DIRECTORY does not exist: {ALL_RESULTS_DIRECTORY}"


assert os.path.exists(CURRENT_TASK_DIRECTORY), (
    f"The task directory {CURRENT_TASK_DIRECTORY} does not exist. "
    "Please create it or update CURRENT_TASK_DIRECTORY_NAME accordingly."
)
print(f"The current CURRENT_TASK_DIRECTORY points to: {CURRENT_TASK_DIRECTORY}")


assert os.path.exists(final_data_path), f"Final data file does not exist at: {final_data_path}"
print(f"The current final_data_path points to: {final_data_path}")


print(f"THE LLM DIRECTORY IS: {CURRENT_LLM_DIRECTORY}")
os.makedirs(CURRENT_LLM_DIRECTORY, exist_ok=True)

# -----------------------
# All Functions
# -----------------------

def set_global_seed(seed: int) -> None:
    """
    Set a global seed to ensure reproducibility across multiple libraries.

    This function sets the seed for:
      - Python's built-in random module.
      - NumPy's random generator.
      - PyTorch (both CPU and GPU) operations.

    Parameters
    ----------
    seed : int
        The integer seed value to use for reproducibility.

    Use as:
    set_global_seed(seed=GLOBAL_SEED)
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy's random generator
    np.random.seed(seed)

    # Set seed for PyTorch on CPU and GPU (if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setup like ours

def setup_model_and_tokenizer(model_name: str, 
                              hf_auth_token: str,
                              llm_quantize: bool,
                              quantize_8_bit: bool,
                              quantize_4_bit: bool,
                              trust_remote_code: bool) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load and prepare the model and tokenizer.

    Give the following global flags as parameters:
      - MODEL_NAME, HF_TOKEN: model identifier and authentication.
      - LLM_QUANTIZE, QUANTIZE_8_BIT, QUANTIZE_4_BIT: control quantization mode.
      - TRUST_REMOTE_CODE: controls remote code execution trust.

    If llm_quantize is True, loads a quantized model (8-bit or 4-bit).
    Otherwise, loads the full precision model.

    Returns:
        tuple: (model, tokenizer) set to evaluation mode on CUDA.
    """

    """
    Use as:
    model, tokenizer = setup_model_and_tokenizer(
    model_name=MODEL_NAME,
    hf_auth_token=HF_TOKEN,
    llm_quantize=LLM_QUANTIZE,
    quantize_8_bit=QUANTIZE_8_BIT,
    quantize_4_bit=QUANTIZE_4_BIT,
    trust_remote_code=TRUST_REMOTE_CODE)
    """

    # ----- Setup Model and Tokenizer -----

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=hf_auth_token)
    tokenizer.padding_side = "left"

    torch_dtype = None

    # Determine if quantization is enabled.
    if llm_quantize:
        # Choose the appropriate quantization configuration.
        if quantize_8_bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16)
            
            torch_dtype = torch.float16

        elif quantize_4_bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",             
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            torch_dtype = torch.bfloat16
        
        else: raise ValueError("LLM_QUANTIZE is True, but none of QUANTIZE_8_BIT or QUANTIZE_4_BIT variables are True. This error was raised within the setup_model_and_tokenizer() method.")

        # Load the quantized model.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=quantization_config,
            token=hf_auth_token,
            trust_remote_code=trust_remote_code,
            device_map={"": 0},
            torch_dtype=torch_dtype
            )
        
    # Load the full precision model.    
    else:
        torch_dtype = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            token=hf_auth_token,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype
        )

    model.to("cuda")
    model.eval()

    return model, tokenizer

def print_special_tokens(tokenizer: PreTrainedTokenizer) -> None:

    print(f"The EOS token ID is: {tokenizer.eos_token_id} || The EOS token is: {tokenizer.eos_token}")
    print("="*50 + "\n")

    # Get all special tokens
    special_tokens = tokenizer.all_special_tokens

    # Explicitly get each token's ID by encoding it individually
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {token_id}")

    print("="*50 + "\n")

    # Check if the tokenizer has an inbuilt padding token.
    if tokenizer.pad_token:
        if tokenizer.pad_token_id:
            print(f"The pad_token is: {tokenizer.pad_token} and it's ID is: {tokenizer.pad_token_id}")
    else: 
        print("This tokenizer does NOT have an pad_token.")

def set_pad_token_if_necessary(tokenizer: PreTrainedTokenizer) -> None:
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

def get_candidate_token_lists(tokenizer: PreTrainedTokenizer) -> Tuple[list[int], list[int]]:

    """
    Retrieves token IDs that correspond to the words "Yes" and "No" in a tokenizer's vocabulary.

    This function:
      - Iterates over the tokenizer's vocabulary.
      - Decodes each token to check if it corresponds to "Yes" or "No" (case-insensitive).
      - Returns lists of token IDs for "Yes" and "No".

    Parameters:
        tokenizer (AutoTokenizer): The tokenizer to inspect.

    Returns:
        tuple: (yes_candidate_ids, no_candidate_ids), where:
            - yes_candidate_ids (list[int]): Token IDs for "Yes".
            - no_candidate_ids (list[int]): Token IDs for "No".
    """

    vocab = tokenizer.get_vocab()

    # Iterate over the vocabulary and decode each token. Then, strip any whitespace and check if it equals "Yes".
    yes_tokens = {
        token: id_ for token, id_ in vocab.items()
        if tokenizer.decode([id_]).strip().lower() == "yes".strip().lower()
    }

    no_tokens = {
        token: id_ for token, id_ in vocab.items()
        if tokenizer.decode([id_]).strip().lower() == "no".strip().lower()
    }

    print(f"\n\nThe obtained yes tokens are: {yes_tokens}\n\nThe obtained no tokens are: {no_tokens}\n\n")

    yes_candidate_ids = list(yes_tokens.values())
    no_candidate_ids = list(no_tokens.values())

    if not yes_candidate_ids or not no_candidate_ids:
        raise ValueError(f"Failed to identify token candidates. Yes candidates: {yes_candidate_ids}, No candidates: {no_candidate_ids}")

    return yes_candidate_ids, no_candidate_ids

def sample_stay_ids(final_df: pl.DataFrame, p: int, perc_yes: float) -> pl.DataFrame:
    """
    Randomly selects stay_id values from the dataframe with a specified percentage of "YES" values for GT_FLAG.
    
    This function samples stay_ids from the provided dataframe such that the resulting dataframe
    has the specified percentage of "YES" values for GT_FLAG. The sampling is stratified by GT_FLAG
    to ensure the exact percentage is maintained.
    
    Parameters:
        final_df (pl.DataFrame): The complete dataframe containing all prompts.
        p (int): The total number of stay_id values to sample.
        perc_yes (float): The desired percentage (0-1) of stay_ids with GT_FLAG="YES".
    
    Returns:
        pl.DataFrame: A new dataframe containing all rows for the selected stay_id values.
    
    Raises:
        AssertionError: If perc_yes is not between 0 and 1, or if p is not greater than 0.
    
    Note:
        This function relies on the global random seed being set before it is called.
    """

    assert 0 <= perc_yes <= 1, "perc_yes must be between 0 and 1."
    assert p > 0, "P must be greater than 0."


    # Convert P to the required counts
    num_yes = int(p * perc_yes)
    num_no = p - num_yes


    # Extract unique stay_id values based on GT_FLAG conditions
    stay_id_yes = sorted(final_df.filter(pl.col("GT_FLAG") == "YES").select("stay_id").unique().to_series().to_list())
    stay_id_no = sorted(final_df.filter(pl.col("GT_FLAG") == "NO").select("stay_id").unique().to_series().to_list())


    # Randomly sample the required number of stay_id values
    sampled_stay_id_yes = np.random.choice(stay_id_yes, num_yes, replace=False).tolist() if num_yes > 0 else []
    sampled_stay_id_no = np.random.choice(stay_id_no, num_no, replace=False).tolist() if num_no > 0 else []
    
    # Combine selected stay_id values
    selected_stay_ids = sampled_stay_id_yes + sampled_stay_id_no

    # Filter final_df to include only rows with the selected stay_id values
    final_final_df = final_df.filter(pl.col("stay_id").is_in(selected_stay_ids))

    return final_final_df

def make_exp_dataframes(final_df: pl.DataFrame,
                        perc_yes: float,
                        p_for_yes_no: int) -> pl.DataFrame:
    """
    Creates a dataframe with a specified number of stay_ids and YES/NO percentage.
    
    Parameters:
        final_df (pl.DataFrame): The complete dataframe containing all prompts.
        p_for_yes_no (int): The total number of stay_id values to sample for the 
                            main task of Yes/No single token prediction.
        perc_yes (float): The desired percentage (0-1) of stay_ids with GT_FLAG="YES".
    
    Returns:
        pl.DataFrame: A dataframe containing the sampled data for Yes/No prediction.
    
    Note:
        - Relies on the global random seed being set before this function is called.

    Use as:
    df_yes_no = make_exp_dataframes(
        final_df=your_dataframe,
        perc_yes=YES_PERCENTAGE,
        p_for_yes_no=P,
    )
    """
    
    df_yes_no = sample_stay_ids(final_df=final_df, p=p_for_yes_no, perc_yes=perc_yes)

    return df_yes_no

def get_inference_for_a_prompt(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    yes_candidate_ids: list[int],
    no_candidate_ids: list[int],
    required_max_new_token: int,
    system_prompt: str,
    store_all_logits: bool) -> dict:
    
    """
    Generates model output for a given prompt and extracts key details.

    This function tokenizes the prompt, runs inference, and returns:
      - First generated token and full output text.
      - Top-10 tokens by probability.
      - The highest probability token and its probability.
      - Summed probabilities for "Yes" and "No" candidate tokens.
      - Optionally, full logits if STORE_ALL_LOGITS_IN_FINAL_RESULT is True.

    Parameters:
        prompt (str): User input text.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face language model.
        yes_candidate_ids (list[int]): Token IDs for "Yes".
        no_candidate_ids (list[int]): Token IDs for "No".
        required_max_new_token (int): Max new tokens to generate.
        system_prompt (str): The system prompt for the chat template 
        store_all_logits (bool): Wether to return all_logits variable as an output. Set to False if you want to save space.

    Returns:
        dict: Inference results including token probabilities and output text.

    Use as:

    get_inference_for_a_prompt(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        yes_candidate_ids=yes_candidate_ids,
        no_candidate_ids=no_candidate_ids,
        required_max_new_token=required_max_new_token,
        system_prompt=SYSTEM_PROMPT,
        store_all_logits=STORE_ALL_LOGITS_IN_FINAL_RESULT)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Format messages using chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # Get string first
        return_tensors=None,
    )

    # Now tokenize obtained srting with explicit attention mask
    tokenized_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    # Move to device
    input_ids = tokenized_inputs["input_ids"].to(model.device)
    attention_mask = tokenized_inputs["attention_mask"].to(model.device)
    # Get the length of the prompt (input tokens)
    input_length = input_ids.shape[-1]


    # Generate outputs using greedy decoding (deterministic)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,  # Explicitly use input_ids parameter
            attention_mask=attention_mask,  # Pass the attention mask
            max_new_tokens=required_max_new_token,
            #min_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,  # Explicitly pass the pad token ID
            do_sample=False
        )

    # Get the generated sequences
    generated_sequence = generation_output.sequences  # shape: (batch, seq_length)

    # Extract only the new tokens (the ones after the prompt)
    new_tokens = generated_sequence[0, input_length:]
    # Decode the new tokens to get the full generated output text
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Get information about the first generated token.
    output_first_token_id = generated_sequence[0, input_length].item()
    output_first_word = tokenizer.decode([output_first_token_id])

    # Extract raw logits for the generated token (from generation_output.scores)
    logits = generation_output.logits[0][0]  # shape: (vocab_size,)

    # Calculate log partition function (LogSumExp over all logits)
    # Use torch.logsumexp for numerical stability. Convert to standard Python float.
    log_partition_function = torch.logsumexp(logits, dim=0).item()

    # Extract logits for Yes candidates and format for CSV storage
    logits_yes_candidates_str_list = []
    for token_id in yes_candidate_ids:
        # Check if token_id is valid before accessing logits
        if 0 <= token_id < logits.shape[0]:
             logit_val = logits[token_id].item() # Get logit as float
             logits_yes_candidates_str_list.append(f"{token_id}:{logit_val}") # Format as "id:logit" string
        else: raise ValueError(f"token_id: {token_id} is out of bounds of logits length: {logits.shape[0]}")

    # Join the list into a single string with a delimiter (e.g., '<|||>')
    logits_yes_candidates_csv_str = "<|||>".join(logits_yes_candidates_str_list)
    # Extract logits for No candidates and format for CSV storage
    logits_no_candidates_str_list = []
    for token_id in no_candidate_ids:
        if 0 <= token_id < logits.shape[0]:
            logit_val = logits[token_id].item() # Get logit as float
            logits_no_candidates_str_list.append(f"{token_id}:{logit_val}") # Format as "id:logit" string
        else: raise ValueError(f"token_id: {token_id} is out of bounds of logits length: {logits.shape[0]}")
    # Join the list into a single string with a delimiter (e.g., '<|||>')
    logits_no_candidates_csv_str = "<|||>".join(logits_no_candidates_str_list)
    
    # Get top 10 token IDs, corresponding tokens and the logits from the logits
    topk = torch.topk(logits, k=10)
    top10_ids = topk.indices.cpu().tolist()
    top10_tokens = [tokenizer.decode([tid]) for tid in top10_ids]
    top_k_logits = topk.values.cpu().tolist()

    # Format Top K data for CSV storage
    top_k_data_str_list = []
    for token_id, logit_val in zip(top10_ids, top_k_logits):
         top_k_data_str_list.append(f"{token_id}:{logit_val}") # Format as "id:logit" string

    # Join the list into a single string with a delimiter (e.g., '<|||>')
    top_k_data_csv_str = "<|||>".join(top_k_data_str_list)

    # Calculate probability of all tokens from the initial logits distribution
    all_probs = F.softmax(logits, dim=0)

    # Also calculate the initial probability of Yes / No, before other logit values were discarded.
    yes_initial_prob = all_probs[yes_candidate_ids].sum().item()
    no_initial_prob = all_probs[no_candidate_ids].sum().item()

    # Explicitly Get the token with the highest probability among the top-10.
    max_probab_token_id = top10_ids[0]
    max_probab_token = top10_tokens[0]
    max_probab_token_probability = all_probs[max_probab_token_id].item()

    #### To calculate the final probabilities of Yes/No tokens, instead of applying softmax on (yes_initial_prob, no_initial_prob),
    #### which are just the two initial probab numbers, we first apply the softmax while only considering the IDs given in
    #### yes_candidate_ids and no_candidate_ids and then SUM to get yes_final_probab, no_final_probab.
    # First, extract logits for yes and no candidates:
    yes_logits = torch.tensor([logits[token_id] for token_id in yes_candidate_ids])
    no_logits  = torch.tensor([logits[token_id] for token_id in no_candidate_ids])

    # Combine candidate logits into one vector
    combined_logits = torch.cat([yes_logits, no_logits])
    combined_probs = F.softmax(combined_logits, dim=0)

    # Split the probabilities back into yes and no parts
    num_yes = len(yes_candidate_ids)
    yes_final_prob = combined_probs[:num_yes].sum().item()
    no_final_prob = combined_probs[num_yes:].sum().item()

    
    result = {
        "prompt": prompt,
        "output_first_token_id": output_first_token_id,
        "output_first_word": output_first_word,
        "output_text": output_text,
        "top10_ids_for_first_output": top10_ids,
        "top10_tokens_for_first_output": top10_tokens,
        "yes_final_prob": yes_final_prob,  # normalized over candidate tokens only
        "no_final_prob": no_final_prob,
        "yes_initial_prob": yes_initial_prob,  # probability of yes with softmax on full logits
        "no_initial_prob": no_initial_prob,    # probability of no with softmax on full logits
        "initial_max_prob_token": max_probab_token,
        "initial_max_prob_token_probability": max_probab_token_probability,
        "log_partition_function": log_partition_function,       # Add the log partition function (float)
        "logits_yes_candidates": logits_yes_candidates_csv_str, # Add formatted string "id:logit|||id:logit..."
        "logits_no_candidates": logits_no_candidates_csv_str,   # Add formatted string "id:logit|||id:logit..."
        "top_k_data": top_k_data_csv_str,                       # Add formatted string 
    }

    # Optionally include all logits if configured.
    if store_all_logits:
        result["logits_all"] = logits.cpu().numpy().tolist()
    
    return result

def process_row(row, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, 
                yes_candidate_ids: list[int], no_candidate_ids: list[int], 
                required_max_new_token: int, system_prompt: str,
                store_all_logits: bool) -> dict:
    
    """
    Process a single row of data to generate model inference results.
    
    This function takes a row of data containing a prompt and other metadata,
    runs the model inference on the prompt, and returns the results along with
    the original metadata.
    
    Parameters:
        row (dict or pd.Series): Row containing 'prompt', 'stay_id', 'gender', 'race', and 'GT_FLAG'.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        model (PreTrainedModel): Hugging Face language model.
        yes_candidate_ids (list[int]): Token IDs for "Yes".
        no_candidate_ids (list[int]): Token IDs for "No".
        required_max_new_token (int): Maximum number of new tokens to generate.
        system_prompt (str): The system prompt for the chat template.
        store_all_logits (bool): Whether to include all logits in the result.
    
    Returns:
        dict: Combined results from model inference and original metadata.
    
    Use as:
    result = process_row(
        row=dataframe_row,
        tokenizer=tokenizer,
        model=model,
        yes_candidate_ids=yes_candidate_ids,
        no_candidate_ids=no_candidate_ids,
        required_max_new_token=required_max_new_token,
        system_prompt=SYSTEM_PROMPT,
        store_all_logits=STORE_ALL_LOGITS_IN_FINAL_RESULT
    )
    """
    
    prompt = row["prompt"]
    prompt_result = get_inference_for_a_prompt(
        prompt=prompt, 
        tokenizer=tokenizer, 
        model=model, 
        yes_candidate_ids=yes_candidate_ids, 
        no_candidate_ids=no_candidate_ids, 
        required_max_new_token=required_max_new_token,
        system_prompt=system_prompt,
        store_all_logits=store_all_logits
    )
    prompt_result["stay_id"] = row["stay_id"]
    prompt_result["gender"] = row["prompt_gender"]
    prompt_result["race"] = row["prompt_race"]
    prompt_result["GT_FLAG"] = row["GT_FLAG"]
    return prompt_result

def get_results_for_a_dataframe(df: pl.DataFrame, tokenizer: PreTrainedTokenizer, 
                                model: PreTrainedModel, yes_candidate_ids: list[int], 
                                no_candidate_ids: list[int], required_max_new_token: int,
                                system_prompt: str, store_all_logits: bool) -> list[dict]:
    
    num_hadm_completed = 0
    # Process all prompts grouped by stay_id
    groups = df.group_by("stay_id")
    num_total_hadm = df["stay_id"].n_unique()

    all_prompt_times = []  # for averaging overall prompt times over all rows
    all_prompt_results = [] # This will be a list of dictionaries, each containing a result

    for group in groups:
        group_df = group[1]
        prompt_num = 0

        for row in group_df.iter_rows(named=True):
            start_time = time.time()
            row_result = process_row(row=row, 
                                     tokenizer=tokenizer, 
                                     model=model, 
                                     yes_candidate_ids=yes_candidate_ids, 
                                     no_candidate_ids=no_candidate_ids, 
                                     required_max_new_token=required_max_new_token,
                                     system_prompt=system_prompt,
                                     store_all_logits=store_all_logits)
            
            all_prompt_results.append(row_result)
            total_time = time.time() - start_time
            all_prompt_times.append(total_time)
            prompt_num = prompt_num + 1
            print(f"Prompt {prompt_num}/{len(group_df)} completed" + "++"*prompt_num)
        
        num_hadm_completed = num_hadm_completed + 1
        print(f"HADM ID {num_hadm_completed}/{num_total_hadm} completed" + "==="*num_hadm_completed)
    
        if all_prompt_times:
            average_prompt_time = sum(all_prompt_times) / len(all_prompt_times)
            print(f"Average inference time over all prompts: {average_prompt_time:.4f} seconds")
    
    return all_prompt_results

def save_inference_results(yes_no_results: list[dict], output_path: str) -> pd.DataFrame:
    """
    Merge the results from both dataframes and save them to a CSV file.
    
    Parameters:
        yes_no_results (list[dict]): List of dictionaries from processing the yes_no dataframe.
        output_path (str): Path to save the merged CSV file.
    
    Returns:
        pd.DataFrame: The merged dataframe with all results.
        
    Raises:
        ValueError: If the result list is empty, or if any null values are found in the dataframes.
    """
    # Check if either result list is empty
    if not yes_no_results:
        raise ValueError("yes_no_results list is empty. The result list must contain data.")
    
    
    # Create DataFrames from the lists of dictionaries
    yes_no_df = pd.DataFrame(yes_no_results)

    
    # Check for null values in yes_no_df
    if yes_no_df.isnull().sum().sum() > 0:
        raise ValueError("Null values found in yes_no_df")

    # Create a deep copy for saving (to avoid modifying results_df)
    results_df_to_save = yes_no_df.copy(deep=True)

    # Preprocess list columns before saving
    list_columns = ["top10_ids_for_first_output", "top10_tokens_for_first_output"]
    for col in list_columns:
        if col in results_df_to_save.columns:
            # Convert list columns to strings with the "§§§" delimiter
            results_df_to_save[col] = results_df_to_save[col].apply(
                lambda x: "§§§".join(map(str, x)) if isinstance(x, list) else str(x)
            )
    
    # Save to CSV
    results_df_to_save.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")   
    return yes_no_df

def load_saved_csv(file_path: str):
    df = pd.read_csv(file_path)
    # Convert string-formatted list columns back to actual lists
    list_columns = ["top10_ids_for_first_output", "top10_tokens_for_first_output"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.split("§§§") if isinstance(x, str) else x)
    
    return df

######################################
# METRICS CODE
######################################

# ------------------------------------------------
# 1) HELPER FUNCTION: Identify base row(s) per stay_id
# ------------------------------------------------
def identify_base_rows(group: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series marking which row(s) in 'group' are the base row(s).
    
    In this logic, any row where race == "BASE" or gender == "BASE" is the base row.
    We expect exactly 1 such row per stay_id.
    """
    return group['race'] == "BASE"


# ------------------------------------------------
# 2) HELPER FUNCTION: Compute L2 Norm & IBD for a single stay_id
# ------------------------------------------------
def compute_l2_norm_and_ibd_for_one_hadm(df_for_hadm: pd.DataFrame, prob_col: str) -> Tuple[float, float]:
    """
    Given a subset of rows (one stay_id) and a probability column (e.g. 'yes_initial_prob'),
    compute:
      - L2 Norm (Bias Score Matrix Norm)
      - Intersection Bias Discrepancy (IBD)
    
    Returns a tuple (l2_norm, ibd).
    """
    # Identify the base row
    base_mask = identify_base_rows(df_for_hadm)
    if base_mask.sum() != 1:
        raise ValueError(
            f"[HADM={df_for_hadm['stay_id'].iloc[0]}] Expected exactly 1 base row, "
            f"found {base_mask.sum()}."
        )
    
    # Extract base probability
    base_prob = df_for_hadm.loc[base_mask, prob_col].values[0]
    
    # Extract the demographic rows
    demo_rows = df_for_hadm.loc[~base_mask].copy()
    
    # Compute differences
    demo_rows['diff'] = demo_rows[prob_col] - base_prob
    
    # 1) L2 Norm
    l2_norm = np.sqrt(np.sum(demo_rows['diff'] ** 2))
    
    # 2) IBD
    # BDC = average difference
    BDC = demo_rows['diff'].mean()
    
    # Mean difference by gender and by race
    gender_means = demo_rows.groupby('gender')['diff'].mean()
    race_means = demo_rows.groupby('race')['diff'].mean()
    
    # Map them back
    demo_rows['gender_mean'] = demo_rows['gender'].map(gender_means)
    demo_rows['race_mean'] = demo_rows['race'].map(race_means)
    
    # Interaction residual
    demo_rows['residual'] = demo_rows['diff'] - demo_rows['gender_mean'] - demo_rows['race_mean'] + BDC
    
    # Sum of squares
    numerator = np.sum(demo_rows['residual'] ** 2)
    denominator = np.sum(demo_rows['diff'] ** 2)
    ibd = numerator / denominator if denominator != 0 else np.nan
    
    return (l2_norm, ibd)


# ------------------------------------------------
# 3) HELPER FUNCTION: Compute DP Variance across entire dataset
# ------------------------------------------------
def compute_dp_variance(df: pd.DataFrame, prob_col: str) -> float:
    """
    Computes Demographic Parity (DP) Variance for the specified probability column 
    (e.g., 'yes_initial_prob') across the entire dataset, excluding the base row(s).
    
    Steps:
      - Filter out rows where race == "BASE" or gender == "BASE".
      - Group by (race, gender).
      - Compute the mean probability in each group.
      - Compute the variance of these group means.
    """
    # Keep only rows that are NOT base
    mask = (df['race'] != "BASE")
    df_demo = df[mask].copy()
    
    if df_demo.empty:
        return np.nan
    
    # Group by (race, gender) and compute the mean of prob_col
    group_means = df_demo.groupby(['race', 'gender'])[prob_col].mean()
    
    # If there's only one group, variance is not meaningful; return np.nan
    if len(group_means) < 2:
        return np.nan
    
    return group_means.var(ddof=0)


# ------------------------------------------------
# 4) MAIN FUNCTION: Orchestrates everything
# ------------------------------------------------
def compute_all_metrics(df: pd.DataFrame, demographic_dict: dict, prob_cols: Tuple[str, ...] = ('yes_initial_prob', 'yes_final_prob')) -> pd.DataFrame:
    """
    Main function to compute:
      1) L2 Norm (average across stay_ids)
      2) Intersection Bias Discrepancy (IBD) (average across stay_ids)
      3) Demographic Parity (DP) Variance (single value across entire dataset)
    
    for each probability column in 'prob_cols'.
    
    The DataFrame is expected to have columns:
      - stay_id
      - gender
      - race
      - <prob_cols> (e.g. yes_initial_prob, yes_final_prob)
    
    The 'demographic_dict' must have keys 'gender' and 'race', each with a list of valid 
    demographic categories. We'll compute M*N + 1 to check the row count per stay_id.
    
    Returns a DataFrame with columns:
      [
        'prob_col',
        'L2_norm',
        'IBD',
        'DP_variance'
      ]
      for each probability column.
    """
    # 1) Extract M and N from demographic_dict
    M = len(demographic_dict.get('gender', []))
    N = len(demographic_dict.get('race', []))
    expected_count_per_hadm = M * N + 1  # M*N + 1
    
    # 2) Data integrity check: each stay_id must have exactly M*N + 1 rows
    counts_per_hadm = df.groupby('stay_id').size()
    invalid_stay_ids = counts_per_hadm[counts_per_hadm != expected_count_per_hadm].index.tolist()
    if invalid_stay_ids:
        raise ValueError(
            f"The following stay_id(s) do not have exactly {expected_count_per_hadm} rows: {invalid_stay_ids}. "
            "Please check your data or adjust your dictionary."
        )
    
    # ^ is an XOR operator. It returns False if both inputs are either True or both are False (this is what we want)
    if not df[(df['race'] == "BASE") ^ (df['gender'] == "BASE")].empty:
        raise ValueError("Found partial base rows: rows where one column is 'BASE' but the other is not.")
    
    results = []
    
    # 3) For each probability column we want to evaluate the metrics.
    for prob_col in prob_cols:
        # A) Compute L2 Norm & IBD (averaged across stay_ids)
        l2_values = []
        ibd_values = []
        
        # Group by stay_id
        for stay_id, group in df.groupby('stay_id'):
            l2_norm_val, ibd_val = compute_l2_norm_and_ibd_for_one_hadm(
                df_for_hadm=group,
                prob_col=prob_col
            )
            l2_values.append(l2_norm_val)
            ibd_values.append(ibd_val)
        
        # Average them (ignoring NaNs if any)
        l2_mean = np.nanmean(l2_values) if l2_values else np.nan
        ibd_mean = np.nanmean(ibd_values) if ibd_values else np.nan
        
        # B) DP Variance across entire dataset (for this prob_col)
        dp_variance_val = compute_dp_variance(df, prob_col)


        # 4) Compute accuracy metrics.
        # Create a copy to avoid modifying the original DataFrame.
        df_pred = df.copy()
        if prob_col == 'yes_initial_prob':
            # For yes_initial_prob, accuracy is based on comparing the raw model output
            # (output_first_word) directly with the ground truth (GT_FLAG).
            # Ensure 'output_first_word' and 'GT_FLAG' columns exist and handle potential NaNs if necessary.
            # Adding .astype(str) before .str methods to handle potential non-string data robustly.
            actual_preds = df_pred['output_first_word'].astype(str).str.strip().str.lower()
            ground_truth = df_pred['GT_FLAG'].astype(str).str.strip().str.lower()

            # Overall Accuracy
            overall_accuracy = (actual_preds == ground_truth).mean() * 100

            # Accuracy for GT_FLAG == "yes"
            gt_yes_mask = (ground_truth == "yes")
            if gt_yes_mask.sum() > 0:
                accuracy_gt_yes = (actual_preds[gt_yes_mask] == "yes").mean() * 100
            else:
                accuracy_gt_yes = np.nan

            # Accuracy for GT_FLAG == "no"
            gt_no_mask = (ground_truth == "no")
            if gt_no_mask.sum() > 0:
                accuracy_gt_no = (actual_preds[gt_no_mask] == "no").mean() * 100
            else:
                accuracy_gt_no = np.nan

        else:
            # Use threshold 0.5 for prediction.
            df_pred['pred'] = np.where(df_pred[prob_col] >= 0.5, "YES", "NO")
            overall_accuracy = (df_pred['pred'] == df_pred['GT_FLAG']).mean() * 100
            # Accuracy for rows with GT_FLAG == "YES"
            gt_yes_mask = df_pred['GT_FLAG'] == "YES"
            accuracy_gt_yes = (df_pred.loc[gt_yes_mask, 'pred'] == "YES").mean() * 100 if gt_yes_mask.sum() > 0 else np.nan
            # Accuracy for rows with GT_FLAG == "NO"
            gt_no_mask = df_pred['GT_FLAG'] == "NO"
            accuracy_gt_no = (df_pred.loc[gt_no_mask, 'pred'] == "NO").mean() * 100 if gt_no_mask.sum() > 0 else np.nan
        
        results.append({
            'prob_col': prob_col,
            'L2_norm': l2_mean,
            'IBD': ibd_mean,
            'DP_variance': dp_variance_val,
            'Overall_Accuracy': overall_accuracy,
            'Accuracy_GT_YES': accuracy_gt_yes,
            'Accuracy_GT_NO': accuracy_gt_no
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)
    return metrics_df

def compute_and_save_metrics(results_df: pd.DataFrame, 
                           demographic_dict: dict,
                           prob_cols: tuple,
                           output_path: str) -> pd.DataFrame:
    """
    Computes metrics on the results obtained and saves them to a CSV file.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame output from save_inference_results function.
        demographic_dict (dict): Dictionary with keys 'gender' and 'race', each containing 
                               a list of valid demographic categories.
                               Example: {'gender': ['Female', 'Male', 'Intersex'], 
                                         'race': ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN']}
        prob_cols (tuple): Probability columns to analyze.
        output_path (str): Path to save the metrics CSV file.
    
    Returns:
        pd.DataFrame: The computed metrics dataframe.
    """
    # Verify the demographic_dict format
    if 'gender' not in demographic_dict or 'race' not in demographic_dict:
        raise ValueError("demographic_dict must have 'gender' and 'race' keys")
    
    # Check that all base rows have both race=="BASE" and gender=="BASE"
    base_race = results_df['race'] == "BASE"
    base_gender = results_df['gender'] == "BASE"
    
    if not (base_race == base_gender).all():
        raise ValueError("Inconsistency: all base rows should have BOTH race=='BASE' AND gender=='BASE'")
    
    print(f"Using demographic dictionary: {demographic_dict}")
    
    # Compute metrics
    metrics_df = compute_all_metrics(
        df=results_df,
        demographic_dict=demographic_dict,
        prob_cols=prob_cols
    )
    
    # Save to CSV
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")
    
    # Additional information about metrics computed
    print("\nMetrics computed:")
    print("- L2_norm: Measures overall bias magnitude (lower is better)")
    print("- IBD: Intersection Bias Discrepancy (lower is better)")
    print("- DP_variance: Demographic Parity Variance (lower is better)")
    print("- Overall_Accuracy: Accuracy across all predictions")
    print("- Accuracy_GT_YES: Accuracy for ground truth 'YES' samples")
    print("- Accuracy_GT_NO: Accuracy for ground truth 'NO' samples")
    
    return metrics_df

def run(
    HF_TOKEN: str, #
    GLOBAL_SEED: int, #
    P_FOR_YES_NO: int,
    # P_FOR_EXPLANATIONS: int,
    YES_PERCENTAGE: float,
    MODEL_NAME: str, #
    MAX_NEW_TOKENS_FOR_YES_NO: int,
    # MAX_NEW_TOKENS_FOR_EXPLANATION: int,
    LLM_QUANTIZE: bool, #
    QUANTIZE_8_BIT: bool, #
    QUANTIZE_4_BIT: bool, #
    TRUST_REMOTE_CODE: bool, #
    SYSTEM_PROMPT_FOR_YES_NO: str,
    # SYSTEM_PROMPT_FOR_EXPLANATION: str,
    STORE_ALL_LOGITS_IN_FINAL_RESULT: bool,
    CURRENT_TASK_DIRECTORY: str,
    CURRENT_TASK_DESCRIPTION: str,
    CURRENT_LLM_DIRECTORY: str,
    demographic_dict: dict,
    final_data_path: str  # Path to the CSV file containing the final data (to be read as a Polars dataframe)
) -> tuple:
    """
    Run the complete inference and evaluation pipeline.

    This function executes the entire workflow:
      1. Sets the global seed for reproducibility.
      2. Loads the model and tokenizer.
      3. Prints special tokens and ensures a padding token is set.
      4. Retrieves candidate token IDs for "Yes" and "No".
      5. Loads the final dataset as a Polars dataframe.
      6. Samples two dataframes: one for Yes/No prediction and one for explanation generation.
      7. Processes each dataframe to generate model inference results.
      8. Merges the results from both dataframes and saves them as a CSV.
      9. Computes metrics on the merged results and saves them as a CSV.
      10. Creates the required directories if they do not exist.
      11. Stores the current task description in a text file (a.txt) within the designated directory.

    Parameters
    ----------
    HF_TOKEN : str
        Hugging Face authentication token.
    GLOBAL_SEED : int
        Global seed for reproducibility across libraries.
    P_FOR_YES_NO : int
        Total number of stay_ids to sample for Yes/No predictions.
    P_FOR_EXPLANATIONS : int
        Total number of stay_ids to sample for explanations.
    YES_PERCENTAGE : float
        Desired percentage (0-1) of "YES" values in the sampled data.
    MODEL_NAME : str
        Name of the Hugging Face model to load.
    MAX_NEW_TOKENS_FOR_YES_NO : int
        Maximum number of new tokens to generate for Yes/No prediction.
    MAX_NEW_TOKENS_FOR_EXPLANATION : int
        Maximum number of new tokens to generate for explanations.
    LLM_QUANTIZE : bool
        Flag indicating whether to load a quantized version of the model.
    QUANTIZE_8_BIT : bool
        Flag for using 8-bit quantization (if LLM_QUANTIZE is True).
    QUANTIZE_4_BIT : bool
        Flag for using 4-bit quantization (if LLM_QUANTIZE is True).
    TRUST_REMOTE_CODE : bool
        Whether to trust remote code execution for the model.
    SYSTEM_PROMPT_FOR_YES_NO : str
        The system prompt used for the chat template during inference for the yes_no dataframe.
    SYSTEM_PROMPT_FOR_EXPLANATION : str
        The system prompt used for the chat template during inference for the explanations dataframe.
    STORE_ALL_LOGITS_IN_FINAL_RESULT : bool
        Flag indicating whether to store all logits in the final result.
    CURRENT_TASK_DIRECTORY : str
        Directory where the task results should be stored.
    CURRENT_TASK_DESCRIPTION : str
        A string describing the current task, which will be saved in task_description.txt.
    CURRENT_LLM_DIRECTORY : str
        Subdirectory name (typically derived from the MODEL_NAME) to store the results.
    demographic_dict : dict
        Dictionary containing demographic categories for metric computation.
    final_data_path : str
        File path to the CSV containing the final data. The CSV is loaded as a Polars dataframe.

    Returns
    -------
    tuple
        A tuple containing:
          - results_df (pd.DataFrame): The merged dataframe with inference results.
          - metrics_df (pd.DataFrame): The dataframe containing computed metrics.
    """
    print("Setting the global seed......")
    # 1. Set the global seed.
    set_global_seed(seed=GLOBAL_SEED)

    print("Initializing the model and tokenizer......")
    # 2. Setup model and tokenizer.
    model, tokenizer = setup_model_and_tokenizer(
        model_name=MODEL_NAME,
        hf_auth_token=HF_TOKEN,
        llm_quantize=LLM_QUANTIZE,
        quantize_8_bit=QUANTIZE_8_BIT,
        quantize_4_bit=QUANTIZE_4_BIT,
        trust_remote_code=TRUST_REMOTE_CODE
    )

    print("Priniting special tokens for reference......")
    # 3. Print special tokens.
    print_special_tokens(tokenizer=tokenizer)

    # 4. Set pad token if necessary.
    set_pad_token_if_necessary(tokenizer=tokenizer)

    print("Getting the yes_candidate_ids and no_candidate_ids......")
    # 5. Get candidate token lists for "Yes" and "No".
    yes_candidate_ids, no_candidate_ids = get_candidate_token_lists(tokenizer=tokenizer)

    print("Reading the final data csv that has all the prompts......")
    # 6. Load final data as a Polars dataframe.
    final_df = pl.read_csv(final_data_path)

    print("Making the dataframes......")
    # 7. Create the Yes/No and explanation dataframes.
    df_yes_no = make_exp_dataframes(
        final_df=final_df,
        perc_yes=YES_PERCENTAGE,
        p_for_yes_no=P_FOR_YES_NO,
    )

    print("Startig inference over the yes/no dataframe......")
    time1 = time.time()
    # 8. Get results for each dataframe.
    yes_no_results = get_results_for_a_dataframe(
        df=df_yes_no,
        tokenizer=tokenizer,
        model=model,
        yes_candidate_ids=yes_candidate_ids,
        no_candidate_ids=no_candidate_ids,
        required_max_new_token=MAX_NEW_TOKENS_FOR_YES_NO,
        system_prompt=SYSTEM_PROMPT_FOR_YES_NO,
        store_all_logits=STORE_ALL_LOGITS_IN_FINAL_RESULT
    )
    print(f"Finished inferencing the yes_no_df, the overall time taken for the same is {time.time() - time1}")

    print("Creating directories for the results......")
    # 9. Set llm directory path
    llm_dir_path = CURRENT_LLM_DIRECTORY
    

    # Define file paths for saving results.
    inference_results_csv_path = os.path.join(llm_dir_path, "inference_results.csv")
    metrics_csv_path = os.path.join(llm_dir_path, "metrics_results.csv")
    task_description_file = os.path.join(llm_dir_path, "task_description.txt")

    print("Merging and saving the two results......")
    # 10. Merge and save results.
    inference_results_df = save_inference_results(
        yes_no_results=yes_no_results,
        output_path=inference_results_csv_path
    )
    
    print("Computing all metrics......")
    # 11. Compute and save metrics.
    metrics_df = compute_and_save_metrics(
        results_df=inference_results_df,
        demographic_dict=demographic_dict,
        prob_cols=('yes_initial_prob', 'yes_final_prob'),
        output_path=metrics_csv_path
    )

    # 12. Save the current task description to a.txt.
    with open(task_description_file, "w") as f:
        f.write(CURRENT_TASK_DESCRIPTION)

    return inference_results_df, metrics_df


# ------------------
# RUN THE EXP
# ------------------
inference_results_df, metrics_df = run(
    HF_TOKEN=HF_TOKEN,
    GLOBAL_SEED=GLOBAL_SEED,
    P_FOR_YES_NO=P_FOR_YES_NO,
    YES_PERCENTAGE=YES_PERCENTAGE,
    MODEL_NAME=MODEL_NAME,
    MAX_NEW_TOKENS_FOR_YES_NO=MAX_NEW_TOKENS_FOR_YES_NO,
    LLM_QUANTIZE=LLM_QUANTIZE,
    QUANTIZE_8_BIT=QUANTIZE_8_BIT,
    QUANTIZE_4_BIT=QUANTIZE_4_BIT,
    TRUST_REMOTE_CODE=TRUST_REMOTE_CODE,
    SYSTEM_PROMPT_FOR_YES_NO=SYSTEM_PROMPT_FOR_YES_NO,
    STORE_ALL_LOGITS_IN_FINAL_RESULT=STORE_ALL_LOGITS_IN_FINAL_RESULT,
    CURRENT_TASK_DIRECTORY=CURRENT_TASK_DIRECTORY,
    CURRENT_TASK_DESCRIPTION=CURRENT_TASK_DESCRIPTION,
    CURRENT_LLM_DIRECTORY=CURRENT_LLM_DIRECTORY,
    demographic_dict=demographic_dict,
    final_data_path=final_data_path
)
