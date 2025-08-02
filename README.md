## Usage

### Installation

```julia
# julia --project="@gafundamentals"
pkg> add https://github.com/sondalex/GAFundamentals.jl.git
pkg> build
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
pkg> precompile
```

```julia
pkg> build
```

### Testing

```julia
pkg> test
```
