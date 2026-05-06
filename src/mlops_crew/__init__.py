"""phishing_email_detection.

A phishing email detection system that can detect whether incoming
email is a phishing email or a regular email.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlops_crew")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "kirtan"
__email__ = "kparekh2@depaul.edu"

__all__ = ["__version__", "__author__", "__email__"]
