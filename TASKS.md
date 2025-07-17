# Tasks

**0. to see what i'm testing here, open a cpu graph like `btop` or task manager etc**

![gif](assets/btop.gif)
> watch the 100% go across all the cores, that's what we're doin!

**1. Install Rust**
```bash
# Install rustup (Rust toolchain manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Configure current shell (or follow the output of the rust installer, it will tell you what to do for misc shells)
source "$HOME/.cargo/env"

# Verify installation
rustc --version 
cargo --version
```

- *Alternative for Windows*: Use [rustup-init.exe](https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe)
- *Troubleshooting*: If encountering SSL errors, try `apt install libssl-dev` (Ubuntu) or `brew install openssl` (macOS)

**2. Install UV (Python package manager)**
```bash
pip install uv
# OR, my prefered way install as standalone binary
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync # in _this_ repo
```

**5. Run Data Collection Script**
```bash
./scripts/run.sh
```

### Post-Execution Cleanup
```bash
mkdir jeremy # use your own name pls.

mv *.png 
# Archive results
zip -r results_$(date +%Y%m%d).zip jeremy/ 

cd ../
rm -rf affinity-testing # you probs don't want my stray code on your machine eh.
```