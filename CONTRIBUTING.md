# Contributing to MS-SLR

Thank you for your interest in contributing to the Motion-Signature Sign Language Recognition project! This document provides guidelines for contributing.

## 🎯 Project Goals

MS-SLR aims to provide accessible, privacy-first sign language recognition for real-world deployment. Contributions should align with these core principles:

1. **Accessibility First**: Solutions that work on commodity hardware
2. **Privacy by Design**: No video storage, local processing preferred
3. **Real-World Viability**: Practical accuracy and latency for actual use
4. **Research Rigor**: Well-documented methodology with reproducible results

## 🛠️ Ways to Contribute

### 1. Bug Reports
Found an issue? Please open a GitHub issue with:
- **System specs**: OS, Python version, hardware
- **Steps to reproduce**: Detailed instructions
- **Expected vs actual behavior**: What should happen vs what does
- **Logs/screenshots**: Any relevant error messages

### 2. Feature Requests
Have an idea? Open an issue with:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Impact assessment**: Who benefits and how?

### 3. Code Contributions
Ready to code? Follow these steps:

#### Setup Development Environment
```bash
# Fork and clone the repository
git clone https://github.com/mohammedhashirfiroze/MS-SLR.git
cd MS-SLR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy
```

#### Coding Standards
- **Style**: Follow PEP 8 (use `black` for formatting)
- **Comments**: Explain *why*, not *what* (code should be self-documenting)
- **Docstrings**: Use Google-style docstrings for functions
- **Type hints**: Add type annotations where helpful
- **Testing**: Include tests for new features

#### Example of Good Code Style
```python
def compute_motion_signature(window: np.ndarray) -> np.ndarray:
    """
    Extract temporal features from hand landmark window.
    
    Computes mean, velocity, and variance over temporal window to capture
    sign dynamics beyond static poses. This is the core innovation enabling
    discrimination of signs with similar hand shapes but different motions.
    
    Args:
        window: (N, 126) array of normalized hand landmarks over N frames
        
    Returns:
        (378,) array: [mean(126), velocity(126), variance(126)]
    """
    mean = window.mean(axis=0)  # Average spatial configuration
    velocity = np.diff(window, axis=0).mean(axis=0)  # Motion dynamics
    variance = window.var(axis=0)  # Movement consistency
    return np.concatenate([mean, velocity, variance])
```

#### Git Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add detailed description of change"

# Push to your fork
git push origin feature/your-feature-name

# Open Pull Request on GitHub
```

#### Commit Message Convention
Use semantic commit messages:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, no logic change)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `perf:` Performance improvements

Examples:
```
feat: add real-time confidence visualization
fix: prevent division by zero in normalization
docs: update installation instructions for Windows
perf: optimize motion signature computation by 40%
```

### 4. Documentation
Help improve docs! Areas needing attention:
- Installation guides for different OS
- Troubleshooting common issues
- Tutorial notebooks
- API documentation
- Translation to other languages

### 5. Testing
Help validate the system:
- Test on different hardware/OS combinations
- Conduct user studies with diverse participants
- Test with different sign language variants
- Benchmark against other systems

## 📊 Research Contributions

If contributing research-related improvements:

1. **Cite relevant work**: Include references to papers/methods used
2. **Provide methodology**: Explain approach and reasoning
3. **Share metrics**: Include quantitative evaluation
4. **Make reproducible**: Provide code, data, and instructions
5. **Document limitations**: Be transparent about scope

## 🧪 Testing Guidelines

Before submitting:

```bash
# Run tests (if implemented)
pytest tests/

# Check code style
black --check .
flake8 .

# Type checking (optional)
mypy main.py
```

Test on actual hardware:
- Run for at least 5 minutes to check stability
- Test with multiple camera configurations
- Verify memory usage stays <500MB
- Confirm latency <100ms

## 📝 Pull Request Process

1. **Update documentation**: README, docstrings, CHANGELOG
2. **Add tests**: For new features or bug fixes
3. **Pass CI checks**: Ensure all automated tests pass
4. **Describe changes**: Clear PR description with motivation
5. **Link issues**: Reference related issues (#123)
6. **Request review**: Tag relevant maintainers

### PR Template
```markdown
## Description
[Clear description of what this PR does]

## Motivation
[Why is this change needed? What problem does it solve?]

## Changes
- Change 1
- Change 2

## Testing
- [ ] Tested on Windows/macOS/Linux
- [ ] Verified no performance regression
- [ ] Added/updated tests
- [ ] Updated documentation

## Screenshots/Demos
[If applicable, add screenshots or demo videos]

## Related Issues
Fixes #123
Related to #456
```

## 🚫 What We Don't Accept

- **Plagiarized code**: Must be original or properly attributed
- **Breaking changes**: Without discussion and migration path
- **Unfocused PRs**: One logical change per PR
- **Undocumented code**: Must include clear comments/docs
- **Performance regressions**: Must maintain <100ms latency
- **Privacy violations**: No features that compromise user data

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions with discussion
- **Discussions**: General questions and ideas

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience
- Nationality, personal appearance
- Race, religion, sexual identity and orientation

### Expected Behavior
- Be respectful and considerate
- Focus on what's best for the project
- Accept constructive criticism gracefully
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment, trolling, or personal attacks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 🙏 Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Academic papers (for significant research contributions)

## 📧 Contact

For questions not suited for public discussion:
- **Name**: Mohammed Hashir Firoze
- **Email**: hashirmuhammed71@gmail.com
---

Thank you for contributing to accessible communication technology! 🌍
