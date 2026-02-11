# Secure Credential Management Guide

Complete guide for securely managing your Hugging Face token without exposing it in plaintext.

---

## Overview

This guide shows how to store your HF_TOKEN securely using **pass** (Unix password manager) with GPG encryption.

**Benefits:**
- ✅ Tokens encrypted with GPG
- ✅ Never stored in plaintext
- ✅ Never exposed in shell history
- ✅ Never visible in process listings
- ✅ Easy to rotate or update
- ✅ Centralized credential management

---

## Quick Start (5 minutes)

### Step 1: Get Your HF Token

1. Visit https://huggingface.co/settings/tokens
2. Create a new token with "Write" permissions
3. Copy the token (save it temporarily)

### Step 2: Store Token Securely

```bash
pass insert huggingface/hf_token
```

You'll be prompted:
```
Enter password for huggingface/hf_token: [paste your token, hidden]
Retype password for huggingface/hf_token: [paste again]
```

✅ Token is now encrypted and stored in `~/.password-store/huggingface/hf_token`

### Step 3: Set Up Shell Integration

```bash
./scripts/setup_hf_credentials.sh
```

This will:
- Check if token is stored ✅
- Add `load_hf_token` function to your shell
- Create secure wrapper script `hf_workflow_secure.sh`

### Step 4: Use Secure Credentials

**Option A: Load token into shell**
```bash
source ~/.bashrc  # or ~/.zshrc
load_hf_token
```

**Option B: Use secure wrapper (recommended)**
```bash
./scripts/hf_workflow_secure.sh
```

**Option C: One-time direct run**
```bash
HF_TOKEN=$(pass show huggingface/hf_token) ./scripts/full_hf_workflow.sh
```

---

## How It Works

### Storage Architecture

```
┌─────────────────────────────────────┐
│  Your HF Token (plaintext)          │
│  hf_xxxxxxxxxxxxxxxxxxxxx           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  GPG Encryption                     │
│  (symmetric encryption with your    │
│   GPG key)                          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  ~/.password-store/                 │
│    huggingface/                     │
│      hf_token.gpg                   │
│  (encrypted binary file)            │
└─────────────────────────────────────┘
```

### Secure Loading Flow

```
1. User runs: ./scripts/hf_workflow_secure.sh

2. Script checks: Is HF_TOKEN already set?
   - If YES: Skip decryption, use existing token
   - If NO: Continue

3. Script calls: pass show huggingface/hf_token
   - pass decrypts the .gpg file using your GPG key
   - Token is temporarily in memory only
   - Never written to disk or files

4. Token exported: export HF_TOKEN=<decrypted_value>
   - Only available in current shell session
   - Lost when shell closes
   - Not persisted anywhere

5. Workflow runs: ./scripts/full_hf_workflow.sh
   - Uses HF_TOKEN from environment
   - Automatically cleared when done
```

---

## Management Commands

### View Token

```bash
# Show masked token (shows first/last few chars)
pass show huggingface/hf_token

# Or use shorter alias (if installed)
hf-login
```

### Edit Token

If you need to update your token:

```bash
pass edit huggingface/hf_token
```

This opens your editor with the current (decrypted) token. Edit and save to update.

### Delete Token

To remove the token (e.g., if compromised):

```bash
pass rm huggingface/hf_token
```

Then you can store a new one:

```bash
pass insert huggingface/hf_token
```

### List All Credentials

```bash
pass ls

# Output:
# Password Store
# └── huggingface
#     └── hf_token
```

---

## Usage Scenarios

### Scenario 1: One-time HF Upload

```bash
# Load token and run workflow in one command
./scripts/hf_workflow_secure.sh
```

Output:
```
✅ HF_TOKEN loaded securely
Creating Hugging Face Repositories
...
Models uploaded successfully!
```

### Scenario 2: Manual Commands

```bash
# Load token
source ~/.bashrc
load_hf_token

# Verify it's loaded
echo "Token loaded: ${HF_TOKEN:0:10}***"

# Create repos manually
./scripts/create_hf_repos.sh

# Train and upload
cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets"
```

### Scenario 3: Scripted Automation

```bash
#!/bin/bash
# my_training_script.sh

# Load credentials securely
export HF_TOKEN=$(pass show huggingface/hf_token)

# Run training
cargo run --release --example mnist_with_hf_upload --features "autodiff,datasets"

# Verify upload
curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/user
```

Then run:
```bash
./my_training_script.sh
```

---

## Security Best Practices

### ✅ DO

- ✅ Store tokens in `pass`
- ✅ Use GPG encryption
- ✅ Keep GPG key secure
- ✅ Rotate tokens regularly
- ✅ Use secure wrapper scripts
- ✅ Load tokens only when needed
- ✅ Clear tokens when shell closes

### ❌ DON'T

- ❌ Store tokens in plaintext
- ❌ Commit tokens to git
- ❌ Put tokens in `.env` files
- ❌ Export tokens in shell profiles
- ❌ Share tokens in emails
- ❌ Log tokens to files
- ❌ Use same token across accounts

---

## Troubleshooting

### "pass: command not found"

Install pass:
```bash
# Debian/Ubuntu
sudo apt-get install pass

# macOS (with Homebrew)
brew install pass

# Fedora/RHEL
sudo dnf install pass
```

### "GPG key not found"

You need to set up GPG first:

```bash
# Initialize pass with your GPG key
pass init your-gpg-key-id

# List your GPG keys
gpg --list-secret-keys

# Create new key if needed
gpg --full-generate-key
```

### "Permission denied" on ~/.password-store

Fix permissions:
```bash
chmod 700 ~/.password-store
chmod 600 ~/.password-store/huggingface/*.gpg
```

### Token Not Decrypting

```bash
# Check if GPG key is accessible
gpg --list-keys

# Test decryption manually
gpg --decrypt ~/.password-store/huggingface/hf_token.gpg

# If GPG key needs passphrase, you'll be prompted
```

### "HF_TOKEN not found in password store"

The token hasn't been stored yet. Store it:
```bash
pass insert huggingface/hf_token
```

---

## Advanced Configuration

### Store Multiple Tokens

```bash
# Store prod token
pass insert huggingface/hf_token_prod

# Store dev token
pass insert huggingface/hf_token_dev

# Load specific token
export HF_TOKEN=$(pass show huggingface/hf_token_prod)
```

### Create Custom Loading Function

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Load HF token with optional environment selection
load_hf_token() {
    local env=${1:-prod}
    local path="huggingface/hf_token_$env"

    if pass show "$path" &>/dev/null; then
        export HF_TOKEN=$(pass show "$path")
        echo "✅ Loaded HF_TOKEN for environment: $env"
    else
        echo "❌ Token not found: $path"
        return 1
    fi
}

# Usage: load_hf_token prod  # or load_hf_token dev
```

### Automatic Loading in New Shells

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Uncomment next line to auto-load on shell startup
# load_hf_token
```

⚠️ **Warning:** Auto-loading may prompt for GPG passphrase on every shell. Use with caution.

---

## Integration with HF Workflow

The credential setup integrates seamlessly with the training workflow:

```bash
# Before (exposed token in history):
export HF_TOKEN="hf_xxxxxxxxxxxxxx"
./scripts/full_hf_workflow.sh

# After (secure credential loading):
./scripts/hf_workflow_secure.sh
```

The secure wrapper automatically:
1. ✅ Checks if token is already in environment
2. ✅ Decrypts token from pass if needed
3. ✅ Loads into HF_TOKEN variable
4. ✅ Runs the workflow
5. ✅ Token cleared when done

---

## Verification

### Verify Token is Secure

```bash
# Check that token is NOT in shell history
grep "hf_" ~/.bash_history ~/.zsh_history 2>/dev/null || echo "✅ Token not in history"

# Check that token is NOT in environment
env | grep -i hf_token || echo "✅ Token not in env (until loaded)"

# Check that pass has it encrypted
ls -la ~/.password-store/huggingface/hf_token.gpg
```

### Test Credential Loading

```bash
# Test loading
pass show huggingface/hf_token | head -c 10
# Should show first 10 chars of token

# Test in workflow
./scripts/hf_workflow_secure.sh --help
# Should work without errors
```

---

## Summary

| Aspect | Plaintext | Secure (pass) |
|--------|-----------|---------------|
| Storage | Visible in files | Encrypted in .gpg |
| History | Logged to shell history | Not logged |
| Processes | Visible in `ps` output | Not visible |
| Management | Manual editing | Easy rotation |
| Automation | Hard to secure | Built-in support |
| Recovery | Lost if compromised | Can rotate anytime |

---

## Getting Help

If you encounter issues:

1. Check troubleshooting section above
2. Verify pass installation: `pass --version`
3. Verify GPG setup: `gpg --list-keys`
4. Test decryption: `pass show huggingface/hf_token`
5. Run setup script again: `./scripts/setup_hf_credentials.sh`

---

**Security Status: ✅ ENHANCED**

Your HF_TOKEN is now encrypted, secure, and never exposed in plaintext!
