# Trading Bot Project Maintenance Guide

## Overview
This guide provides instructions for maintaining the organized project structure and code quality established during the comprehensive cleanup of September 25, 2025.

## Code Quality Standards

### Lint Checking
```bash
# Check code quality
python3 -m flake8 trading_bot/ --statistics --count

# Target: Keep errors under 50 (currently 61)
# Focus on: F401 (unused imports), W292 (newlines), F541 (f-strings)
```

### Auto-Formatting
```bash
# Format code with black
python3 -m black trading_bot/ --line-length 88

# Sort imports
python3 -m isort trading_bot/ --profile black

# Run both with the utility script
python3 claudedocs/comprehensive_lint_fix.py
```

## Directory Organization Rules

### Source Code Structure
**MAINTAIN THIS STRUCTURE** - Do not mix modules:
```
trading_bot/
├── core/                # Configuration, logging, events
├── market_data/         # Data sources and processing
├── strategies/          # Trading strategies
├── risk_management/     # Risk assessment
├── execution/           # Order execution
├── portfolio_manager/   # Portfolio management
└── notification/        # Discord notifications
```

### Test Organization
**MIRROR SOURCE STRUCTURE** in tests:
```
tests/
├── core/               # Tests for core modules
├── market_data/        # Tests for market data modules
├── execution/          # Tests for execution engine
├── strategies/         # Tests for trading strategies
├── notification/       # Tests for notifications
├── portfolio_manager/  # Tests for portfolio management
├── risk_management/    # Tests for risk management
└── test_integration.py # Integration tests
```

**Rules**:
- New module `trading_bot/X/` → Create `tests/X/`
- Always include `__init__.py` in test directories
- Name tests as `test_module_name.py`

### Development Files Organization
**KEEP ORGANIZED** in dev-files:
```
dev-files/
├── demos/              # Demo scripts (*_demo.py, demo_*.py)
├── examples/           # Usage examples (example_*.py, *_example.py)
├── integration_tests/  # Integration testing scripts
├── utility_scripts/    # Maintenance and utility scripts
├── docs/              # Development documentation
├── *.md              # Documentation files (keep in root)
```

**Rules**:
- Demo files → `dev-files/demos/`
- Example files → `dev-files/examples/`
- Test integration → `dev-files/integration_tests/`
- Scripts/utilities → `dev-files/utility_scripts/`
- Never put dev files in project root or main source tree

## Adding New Components

### New Trading Strategy
1. Create source: `trading_bot/strategies/new_strategy.py`
2. Create test: `tests/strategies/test_new_strategy.py`
3. Add demo: `dev-files/demos/new_strategy_demo.py`
4. Update imports and maintain code quality

### New Risk Management Module
1. Create source: `trading_bot/risk_management/new_module.py`
2. Create test: `tests/risk_management/test_new_module.py`
3. Follow existing patterns for dependencies and interfaces

### New Notification Channel
1. Create source: `trading_bot/notification/new_notifier.py`
2. Create test: `tests/notification/test_new_notifier.py`
3. Add integration example: `dev-files/examples/new_notifier_example.py`

## Quality Gates

### Before Committing
```bash
# 1. Check lint status
python3 -m flake8 trading_bot/ --count

# 2. Run tests in affected areas
python3 -m pytest tests/specific_module/ -v

# 3. Format code
python3 -m black trading_bot/
python3 -m isort trading_bot/ --profile black
```

### Monthly Maintenance
```bash
# 1. Full lint check and fix
python3 claudedocs/comprehensive_lint_fix.py

# 2. Check for unused imports
python3 -c "import subprocess; subprocess.run(['flake8', 'trading_bot/', '--select=F401'])"

# 3. Verify test structure matches source
find trading_bot/ -name "*.py" -not -path "*/__pycache__/*" | sed 's/trading_bot/tests/' | sed 's/\.py$//' | while read test_dir; do [ -d "$(dirname "$test_dir")" ] || echo "Missing test directory: $(dirname "$test_dir")"; done
```

## Common Issues and Solutions

### Lint Error: F401 (Unused Import)
**Problem**: Import statements not being used
**Solution**:
```python
# Remove unused imports or use them
# Use tools to detect: python3 -m flake8 --select=F401
```

### Lint Error: E501 (Line Too Long)
**Problem**: Lines exceed 88 characters
**Solutions**:
```python
# Option 1: Break at logical points
very_long_function_call(
    parameter1="value1",
    parameter2="value2",
    parameter3="value3"
)

# Option 2: Use intermediate variables
intermediate_result = some_complex_calculation()
final_result = another_function(intermediate_result)
```

### Missing Test Directory
**Problem**: New module added without corresponding test directory
**Solution**:
```bash
# Create test directory structure
mkdir -p tests/new_module/
touch tests/new_module/__init__.py
touch tests/new_module/test_new_module.py
```

### Scattered Development Files
**Problem**: Demo/example files in wrong locations
**Solution**:
```bash
# Move to proper location
mv misplaced_demo.py dev-files/demos/
mv example_usage.py dev-files/examples/
mv integration_test.py dev-files/integration_tests/
```

## File Naming Conventions

### Source Files
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase` within files
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### Test Files
- **Pattern**: `test_module_name.py`
- **Classes**: `TestClassName`
- **Methods**: `test_method_name`

### Development Files
- **Demos**: `demo_feature.py` or `feature_demo.py`
- **Examples**: `example_usage.py` or `feature_example.py`
- **Integration**: `test_feature_integration.py`
- **Utilities**: `fix_*.py`, `utility_*.py`, `*_script.py`

## Documentation Standards

### Code Documentation
- **Modules**: Include docstring explaining purpose
- **Functions**: Google-style docstrings with Args/Returns
- **Classes**: Purpose and usage examples

### Project Documentation
- **Changes**: Document in claudedocs/ for significant modifications
- **Architecture**: Update when adding new major components
- **Maintenance**: Keep this guide updated

## Monitoring and Alerts

### Quality Metrics to Track
- Lint error count (target: < 50)
- Test coverage by module
- Import dependency complexity
- File organization compliance

### Warning Signs
- Lint errors increasing beyond 75
- Test files not matching source structure
- Development files scattered in main directories
- Missing documentation for new components

## Tool Configuration

### VS Code Settings (Recommended)
```json
{
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.linting.flake8Args": ["--max-line-length=88"]
}
```

### Pre-commit Hook (Optional)
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        args: [--line-length=88]
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
```

## Emergency Procedures

### Project Structure Corruption
1. **Assessment**: Run structure validation script
2. **Backup**: Ensure git commits are clean
3. **Restoration**: Use organization patterns from this guide
4. **Verification**: Run full lint and test suite

### Major Refactoring
1. **Planning**: Document intended changes
2. **Testing**: Ensure comprehensive test coverage
3. **Incremental**: Make changes module by module
4. **Validation**: Maintain quality gates throughout

---

**Last Updated**: September 25, 2025
**Next Review**: October 25, 2025
**Maintainer**: Project Team