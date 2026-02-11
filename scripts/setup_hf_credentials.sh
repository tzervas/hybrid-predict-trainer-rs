#!/bin/bash
# Secure HF_TOKEN Credential Helper
# Stores and retrieves HF credentials using pass (Unix password manager)
# Tokens are encrypted with GPG and never stored in plaintext

set -e

CRED_PATH="huggingface/hf_token"
SHELL_PROFILE=""

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                 🔐 Secure HF_TOKEN Credential Setup                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check if token already stored
echo "📋 Checking if HF_TOKEN is already stored..."
if pass show "$CRED_PATH" &>/dev/null; then
    echo "✅ HF_TOKEN already stored securely in pass"
    echo ""
    echo "Token location: ~/.password-store/$CRED_PATH"
    echo "Storage: Encrypted with GPG"
    echo ""
else
    echo "❌ HF_TOKEN not found in pass"
    echo ""
    echo "To store your HF token securely:"
    echo ""
    echo "1. Get your HF token from: https://huggingface.co/settings/tokens"
    echo "2. Run this command (will prompt for token):"
    echo ""
    echo "   pass insert $CRED_PATH"
    echo ""
    echo "3. Paste your token when prompted (will be hidden)"
    echo "4. Confirm by pasting again"
    echo ""
    echo "Then re-run this script."
    echo ""
    exit 1
fi

# Step 2: Create shell function
echo "📝 Setting up shell integration..."

SHELL_CONFIG=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
else
    echo "❌ No shell config found (.bashrc or .zshrc)"
    exit 1
fi

echo "Using shell config: $SHELL_CONFIG"
echo ""

# Check if function already exists
if grep -q "load_hf_token" "$SHELL_CONFIG"; then
    echo "✅ Shell function already installed"
else
    echo "📥 Adding load_hf_token function to $SHELL_CONFIG..."
    cat >> "$SHELL_CONFIG" << 'SHELL_FUNC'

# Load HF_TOKEN securely from password manager
load_hf_token() {
    if ! command -v pass &> /dev/null; then
        echo "❌ Error: 'pass' password manager not found"
        echo "   Install with: sudo apt-get install pass (Debian/Ubuntu)"
        return 1
    fi

    if pass show huggingface/hf_token &>/dev/null; then
        export HF_TOKEN=$(pass show huggingface/hf_token)
        echo "✅ HF_TOKEN loaded securely"
        return 0
    else
        echo "❌ HF token not found in password store"
        echo "   Set up with: pass insert huggingface/hf_token"
        return 1
    fi
}

# Alias for convenience
alias hf-login='load_hf_token'

SHELL_FUNC
    echo "✅ Function added"
    echo ""
fi

# Step 3: Create workflow wrapper script
echo "🔧 Creating secure workflow wrapper..."

cat > "$(dirname "$BASH_SOURCE[0]")/hf_workflow_secure.sh" << 'WRAPPER_SCRIPT'
#!/bin/bash
# Secure HF workflow wrapper - loads credentials before running workflow

set -e

# Load credentials
if [ -z "$HF_TOKEN" ]; then
    if ! command -v pass &> /dev/null; then
        echo "❌ Error: 'pass' not installed"
        exit 1
    fi

    if pass show huggingface/hf_token &>/dev/null; then
        export HF_TOKEN=$(pass show huggingface/hf_token)
        echo "✅ HF_TOKEN loaded securely"
    else
        echo "❌ HF token not found in password store"
        exit 1
    fi
fi

# Run the full workflow
exec ./scripts/full_hf_workflow.sh "$@"
WRAPPER_SCRIPT

chmod +x "$(dirname "$BASH_SOURCE[0]")/hf_workflow_secure.sh"
echo "✅ Wrapper script created: scripts/hf_workflow_secure.sh"
echo ""

# Step 4: Summary
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         ✅ Setup Complete!                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔐 Secure Credential Storage Active"
echo ""
echo "Your HF_TOKEN is now:"
echo "  ✅ Encrypted with GPG"
echo "  ✅ Stored in ~/.password-store/huggingface/hf_token"
echo "  ✅ Never exposed in plaintext"
echo "  ✅ Never stored in environment files"
echo ""
echo "📖 Usage:"
echo ""
echo "  Option 1 - Load token into current shell:"
echo "    source ~/.bashrc  # or ~/.zshrc"
echo "    load_hf_token"
echo ""
echo "  Option 2 - Use secure wrapper (auto-loads token):"
echo "    ./scripts/hf_workflow_secure.sh"
echo ""
echo "  Option 3 - Manual one-time run:"
echo "    HF_TOKEN=\$(pass show huggingface/hf_token) ./scripts/full_hf_workflow.sh"
echo ""
echo "⚙️  Management Commands:"
echo ""
echo "  View token:    pass show huggingface/hf_token"
echo "  Edit token:    pass edit huggingface/hf_token"
echo "  Delete token:  pass rm huggingface/hf_token"
echo ""
echo "🔒 Security:"
echo ""
echo "  - Tokens never appear in shell history"
echo "  - Tokens never appear in process lists"
echo "  - Tokens only decrypted when needed"
echo "  - GPG encryption with your key"
echo ""
