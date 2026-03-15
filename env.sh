# Description: Set up environment for running experiments. e.g. source env.sh 0
# Arguments: $1: GPU ID

if [ -z "$1" ]
then
    echo "Usage: source env.sh <GPU ID>"
    return
fi

# function load_env to export environment variables from function argument file
function load_env() {
    if [ -f "$1" ]; then
        # shellcheck disable=SC2046
        export $(grep -v '^#' "$1" | xargs)
    else
        echo "Error: $1 not found"
         return
    fi
}

function assert_not_empty() {
    if [ -z "$1" ]; then
        echo "Error: $2"
        return
    fi
}

echo "Using GPU $1"
export CUDA_VISIBLE_DEVICES=$1

# set environment
load_env "env/cache_dir.env"
load_env "env/online.env"
load_env "env/project_config.env"

assert_not_empty "$CONDA_DIR" "CONDA_DIR is not set"
assert_not_empty "$CONDA_ENV_NAME" "CONDA_ENV_NAME is not set"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# if pip installed hf_transfer, enable it
if pip list | grep "hf_transfer"; then
    echo "hf_transfer is installed, enabling it"
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi