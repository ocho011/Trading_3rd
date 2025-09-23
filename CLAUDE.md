# Claude Code Instructions

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md

# Coding Principles and Guidelines

## SOLID Principles Adherence (Mandatory)

* **Single Responsibility Principle (SRP)**: Each class should have only one clear responsibility
* **Open-Closed Principle (OCP)**: Design structure to be open for extension and closed for modification
* **Liskov Substitution Principle (LSP)**: Implement so that objects of supertype can be replaced with objects of subtype
* **Interface Segregation Principle (ISP)**: Provide client-specific specialized interfaces
* **Dependency Inversion Principle (DIP)**: Depend on abstractions, not concrete implementations

## Code Quality Standards

* **Readability**: Clear and intuitive code structure, write self-evident code to minimize comments
* **Extensibility**: Minimize existing code modifications when adding new strategies or features
* **Simplicity**: Remove unnecessary complexity, focus on core logic
* **Reusability**: Modularize common functionality, eliminate duplicate code

## Naming Conventions (PEP 8 Compliance Mandatory)

* Class names: PascalCase (e.g., UserManager, DataProcessor)
* Function/method names: snake_case (e.g., get_user_data, process_request)
* Variable names: snake_case (e.g., user_id, total_count)
* Constant names: UPPER_SNAKE_CASE (e.g., MAX_RETRY_COUNT, DEFAULT_TIMEOUT)

## Mandatory Verification Items When Writing Code

1. Write each function within 20 lines
2. Verify that each class has a single responsibility
3. Apply dependency injection pattern
4. Mandatory use of type hints
5. Write docstrings (Google style)

## Code Quality
* **Lint Check**: `python3 -m flake8 application/ domain/ infrastructure/ tests/ main.py` - Check code style and errors
* **Format Code**: `python3 -m black .` - Auto-format Python code
* **Sort Imports**: `python3 -m isort .` - Organize import statements
* **Full Lint**: `python3 -m pylint application/ domain/ infrastructure/` - Comprehensive 

## Workflow for Code development and refactoring
- **MANDATORY**: Files that are only needed, used, or generated during the development phase should be managed in separate directories (`dev-files/`). This includes:
  - Demo scripts (`demo*.py`, `example*.py`)
  - Development documentation (except README.md, CLAUDE.md)
  - Example configurations and test data
  - Temporary files created during development
- **MANDATORY**: When generating code, use the code-writer subagent, and when reviewing the generated code, use the code-reviewer subagent respectively.
- Use `/organize-dev-files` command regularly to maintain clean project structure.