# Organize Development Files

Automatically organize development phase files according to project guidelines.

## Usage
Use this command whenever creating demo files, examples, or development documentation.

## Steps

1. Check if any files in the root directory are development-only files
2. Move demo files, examples, and development docs to `dev-files/`
3. Update `.gitignore` if needed to exclude development files
4. Verify the root directory only contains production-relevant files

## Development File Types to Move
- `demo*.py` - Demo scripts
- `example*.py` - Example scripts
- `*_example.*` - Example files
- `*.md` files that are development summaries/docs (not README.md or CLAUDE.md)
- `examples/` directory contents
- `docs/` directory if it contains only development docs
- Test data files
- Temporary configuration files

## Production Files to Keep in Root
- Main application code (`trading_bot/`)
- Tests (`tests/`)
- Configuration templates (`.env.example`, `config.ini.example`)
- Project documentation (`README.md`, `CLAUDE.md`)
- Requirements and dependencies (`requirements.txt`, `pyproject.toml`)
- Git and editor configurations