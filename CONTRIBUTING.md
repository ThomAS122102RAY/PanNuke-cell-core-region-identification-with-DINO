# Contributing to PanNuke-cell-core-region-identification-with-DINO

Thank you for your interest in contributing to this project! 🎉

We welcome contributions from the community to help improve this nuclei segmentation pipeline. Whether you're fixing bugs, adding features, improving documentation, or suggesting enhancements, your contributions are greatly appreciated.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## 📜 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## 🤝 How Can I Contribute?

### 1. Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Detailed steps to reproduce the problem
- Expected vs. actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant error messages or logs

### 2. Suggesting Enhancements

We welcome feature requests! Please create an issue describing:
- The motivation for the enhancement
- How it would improve the project
- Any implementation ideas you have

### 3. Code Contributions

You can contribute by:
- Fixing bugs
- Implementing new features
- Improving documentation
- Optimizing performance
- Adding tests
- Improving visualization tools

## 🛠️ Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PanNuke-cell-core-region-identification-with-DINO.git
   cd PanNuke-cell-core-region-identification-with-DINO
   ```

3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Or for bug fixes:
   ```bash
   git checkout -b fix/bug-description
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Make your changes** and test them thoroughly

6. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

7. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** from your fork to the main repository

## 🔄 Pull Request Process

1. **Update Documentation**: If you've added features, update the README.md and any relevant documentation.

2. **Test Your Changes**: Ensure your code works as expected and doesn't break existing functionality.

3. **Follow Code Style**: Adhere to the project's coding style (see Style Guidelines below).

4. **Write Clear Commit Messages**: Use descriptive commit messages that explain what and why.

5. **Keep PRs Focused**: Submit separate PRs for different features or fixes.

6. **Be Responsive**: Be prepared to respond to review comments and make requested changes.

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated (if applicable)
- [ ] No new warnings generated
- [ ] Tested on supported Python versions (3.8+)
- [ ] Tested with PyTorch 1.9+

## 🎨 Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

### Code Example:

```python
def process_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process an input image for model inference.
    
    Args:
        image: Input image as numpy array (H, W, C)
        normalize: Whether to normalize the image
        
    Returns:
        Processed image ready for inference
    """
    if normalize:
        image = image.astype(np.float32) / 255.0
    return image
```

### Commit Message Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when applicable

Examples:
```
Add dual-color overlay visualization
Fix data loading error with prefixed filenames
Update README with installation instructions
```

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Environment Information:**
   - Operating System (Windows/Linux/macOS)
   - Python version
   - PyTorch version
   - CUDA version (if using GPU)

2. **Steps to Reproduce:**
   - Detailed step-by-step instructions
   - Sample data (if applicable)
   - Configuration used

3. **Expected Behavior:**
   - What you expected to happen

4. **Actual Behavior:**
   - What actually happened
   - Error messages or logs

5. **Additional Context:**
   - Screenshots (if applicable)
   - Any workarounds you've tried

## 💡 Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

1. **Use a clear and descriptive title**
2. **Provide a detailed description** of the proposed enhancement
3. **Explain why this enhancement would be useful**
4. **List any alternatives** you've considered
5. **Include mockups or examples** if applicable

## 🏆 Recognition

Contributors will be acknowledged in the project. Significant contributions may be highlighted in release notes.

## 📞 Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Check existing issues to see if your question has been answered

---

Thank you for contributing to PanNuke-cell-core-region-identification-with-DINO! 🙏

Your efforts help make this project better for everyone in the medical imaging and deep learning community.
