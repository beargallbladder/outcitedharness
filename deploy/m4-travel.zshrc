# Sam Kim Dev Env
export DEV="$HOME/Dev"
export PATH="$HOME/.local/bin:$PATH"
if command -v zoxide >/dev/null 2>&1; then
  eval "$(zoxide init zsh)"
fi
alias gs="git status"
alias ga="git add"
alias gc="git commit -m"
alias gp="git push"
alias dev="cd $DEV"
alias c="cursor ."
alias cchat="claude chat"
alias c1="claude complete --model claude-3-haiku"
alias c2="claude complete --model claude-3-sonnet"
alias c3="claude complete --model claude-3-opus"

export NVM_DIR="$HOME/.nvm"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  source "$NVM_DIR/nvm.sh"
elif [[ -s "/opt/homebrew/opt/nvm/nvm.sh" ]]; then
  source "/opt/homebrew/opt/nvm/nvm.sh"
fi
[[ -d "$HOME/.nvm/versions/node/v22.17.0/bin" ]] &&
  export PATH="$HOME/.nvm/versions/node/v22.17.0/bin:$PATH"

export PATH="$HOME/.opencode/bin:$PATH"
export PATH="/opt/homebrew/opt/libpq/bin:$PATH"
if [[ -o interactive && -t 0 ]]; then
  stty -ixon
fi
setopt interactive_comments
