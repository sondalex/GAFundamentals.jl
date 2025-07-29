## Usage

### Installation

```julia
# julia --project="@gafundamentals"
add https://github.com/sondalex/GAFundamentals.jl.git
```
### Command Line

If not set, add ~/.julia/bin to your path:
```zsh
# .zshrc
JULIA_INSTALL="$HOME/.julia"
export PATH="$JULIA_INSTALL/bin:$PATH"
```

```bash
gafundamentals --help
```

## Development

```julia
] precompile
```

```julia
] build
```
