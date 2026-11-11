# Contributing to Reflectorch

## 1. Reporting Issues

If you encounter bugs, want to suggest features, or have questions, please open an issue using the [GitHub issue tracker](https://github.com/schreiber-lab/reflectorch/issues).

When reporting a bug, please include:
- A clear description of the issue
- Steps to reproduce the bug (if applicable)
- Any relevant error messages or screenshots

## 2. Contributing Code

To contribute code:

1. **Fork** the repository.
2. Make your changes in a new branch.
3. Ensure your code is clear and well-documented.
4. If applicable, add or update tests.
5. Submit a [pull request](https://github.com/schreiber-lab/reflectorch/pulls) with a description of your changes.

### Documentation

If your changes affect user-facing features, please update the documentation accordingly. Documentation is built using [Jupyter Book](https://jupyterbook.org/). You can build it locally using:

```bash
jupyter-book build documentation
```

### PyPi releases

The repository contains a Github action for automatic publishing to PyPi via Trusted Publishers (https://docs.pypi.org/trusted-publishers/). The publishing workflow will be triggered automatically upon release of a new Github tag. Additionally, the publishing workflow can be triggered manually from the "Actions" tab. The action is completed after the approval from a main developer.