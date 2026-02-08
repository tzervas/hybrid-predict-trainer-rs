#!/bin/bash
# Create Hugging Face repositories for model variants
# Requires HF_TOKEN environment variable

set -e

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set"
    echo ""
    echo "To get your token:"
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a new token with 'write' permissions"
    echo "  3. Export it: export HF_TOKEN='your_token_here'"
    echo ""
    exit 1
fi

USERNAME="tzervas"

echo "======================================================================="
echo "📦 Creating Hugging Face Model Repositories"
echo "======================================================================="
echo ""
echo "Username: $USERNAME"
echo "Token: ${HF_TOKEN:0:8}..." # Show first 8 chars
echo ""

# Define repositories
declare -a REPOS=(
    "mnist-cnn-traditional:MNIST CNN trained with traditional SGD"
    "mnist-cnn-hybrid:MNIST CNN trained with hybrid predictive method"
    "mnist-cnn-hybrid-gpu:MNIST CNN with GPU-accelerated hybrid training"
    "gpt2-small-traditional:GPT-2 Small (124M) traditional training baseline"
    "gpt2-small-hybrid:GPT-2 Small (124M) with hybrid predictive training"
)

# Function to create repository
create_repo() {
    local repo_name=$1
    local description=$2
    local full_name="$USERNAME/$repo_name"

    echo "Creating repository: $full_name"
    echo "  Description: $description"

    # Use curl to create repo via HF API
    response=$(curl -s -X POST \
        "https://huggingface.co/api/repos/create" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"type\": \"model\",
            \"name\": \"$repo_name\",
            \"organization\": \"$USERNAME\",
            \"private\": false
        }")

    # Check response
    if echo "$response" | grep -q '"error"'; then
        if echo "$response" | grep -q "already exists"; then
            echo "  ✅ Repository already exists"
        else
            echo "  ❌ Error: $response"
        fi
    else
        echo "  ✅ Created successfully"
    fi

    echo ""
}

# Create each repository
for repo_spec in "${REPOS[@]}"; do
    IFS=':' read -r repo_name description <<< "$repo_spec"
    create_repo "$repo_name" "$description"
    sleep 1  # Rate limiting
done

echo "======================================================================="
echo "✅ Repository Creation Complete"
echo "======================================================================="
echo ""
echo "Created repositories:"
for repo_spec in "${REPOS[@]}"; do
    IFS=':' read -r repo_name description <<< "$repo_spec"
    echo "  - https://huggingface.co/$USERNAME/$repo_name"
done
echo ""
echo "Next steps:"
echo "  1. Run training: cargo run --release --example mnist_with_hf_upload --features 'autodiff,datasets'"
echo "  2. Models will be automatically uploaded to repositories"
echo "  3. Check your profile: https://huggingface.co/$USERNAME"
echo ""
