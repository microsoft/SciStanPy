#!/usr/bin/env python3
"""
Azure-specific credential scanner for pre-commit hooks.
Detects Azure credentials, secrets, and sensitive information with minimal false positives.
"""

import os
import re
import sys

# Refined Azure-specific patterns
AZURE_PATTERNS = {
    "Azure Storage Account Key": r"(?i)(?:DefaultEndpointsProtocol=https?;AccountName=[^;]+;AccountKey=|AccountKey=)[A-Za-z0-9+/=]{80,}(?==?)",
    "Azure Storage Connection String": r"(?i)DefaultEndpointsProtocol=https?;AccountName=[a-zA-Z0-9]+;AccountKey=[A-Za-z0-9+/=]{80,}(?==?)",
    "Azure Service Principal Secret": r'(?i)(?:client_secret|password)["\':\s=]+["\']?[a-zA-Z0-9~._-]{34,40}["\']?',
    "Azure DevOps PAT": r'(?i)(?:pat|personal_access_token|token)["\':\s=]+["\']?[a-zA-Z0-9]{52}["\']?',
    "Azure Key Vault Secret URL": r"https://[a-zA-Z0-9-]+\.vault\.azure\.net/secrets/[a-zA-Z0-9-]+",
    "Azure Resource Manager Token": r"Bearer [a-zA-Z0-9+/=]{500,}",
    "Azure AD Token (JWT)": r"eyJ[a-zA-Z0-9+/=]*\.eyJ[a-zA-Z0-9+/=]*\.[a-zA-Z0-9+/=_-]*",
    "Azure Function Key": r'(?i)(?:function_key|code)["\':\s=]+["\']?[a-zA-Z0-9+/=]{44}(?:==)?["\']?',
    "Azure Cosmos DB Key": r'(?i)(?:cosmos_key|primarykey)["\':\s=]+["\']?[a-zA-Z0-9+/=]{88}(?:==)?["\']?',
    "Azure SAS Token": r"(\?|&)sv=\d{4}-\d{2}-\d{2}&[a-zA-Z0-9&=%]+",
}

# Common false positive indicators
FALSE_POSITIVE_INDICATORS = [
    "example",
    "placeholder",
    "your_",
    "todo",
    "fixme",
    "$",
    "{{",
    "}}",
    "<",
    ">",
    "template",
    "dummy",
    "test",
    "mock",
    "fake",
    "sample",
    "00000000-0000-0000-0000-000000000000",
    "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "docs",
    "_sources",
    "__pycache__",
    ".pyc",
    "api/",
    ".git/",
    "# pragma: allowlist secret",
]

# File extensions to skip
SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
}


def is_false_positive(match_text, line_content, filepath):
    """Check if a match is likely a false positive."""
    # Skip binary files
    if any(filepath.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True

    # Skip documentation and generated files
    if any(
        indicator in filepath.lower()
        for indicator in ["docs/", "_sources/", "__pycache__/"]
    ):
        return True

    # Check for allowlist comment
    if "# pragma: allowlist secret" in line_content:
        return True

    # Check for common false positive indicators
    text_to_check = (match_text + " " + line_content).lower()
    return any(indicator in text_to_check for indicator in FALSE_POSITIVE_INDICATORS)


def get_line_number(content, match_start):
    """Get line number for a match position."""
    return content[:match_start].count("\n") + 1


def scan_file(filepath):
    """Scan a single file for Azure credentials."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        secrets_found = False

        for secret_type, pattern in AZURE_PATTERNS.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get the line containing the match
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.end())
                if line_end == -1:
                    line_end = len(content)
                line = content[line_start:line_end].strip()

                # Check for false positives using our enhanced function
                if is_false_positive(match.group(), line, filepath):
                    continue

                # Get line number
                line_num = get_line_number(content, match.start())

                print(f"🚨 AZURE SECRET DETECTED: {secret_type}")
                print(f"   File: {filepath}:{line_num}")
                print(
                    f"   Matched: {match.group()[:50]}{'...' if len(match.group()) > 50 else ''}"
                )
                print(f"   Line: {line}")
                print()
                secrets_found = True

        return secrets_found

    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not scan {filepath}: {e}")
        return False


def main():
    """Main function to scan files provided as command line arguments."""
    files_to_scan = sys.argv[1:] if len(sys.argv) > 1 else []
    if not files_to_scan:
        print("✅ No files to scan.")
        return 0

    secrets_found = False
    for filepath in files_to_scan:
        if os.path.isfile(filepath) and scan_file(filepath):
            secrets_found = True

    if secrets_found:
        print(
            "❌ Azure credentials or secrets detected! Please remove them before committing."
        )
        print()
        print("💡 Consider using:")
        print("   - Azure Key Vault for secrets")
        print("   - Environment variables for configuration")
        print("   - Azure Managed Identity for authentication")
        print("   - Azure App Configuration for settings")
        print("   - Add # pragma: allowlist secret comment for false positives")
        return 1
    else:
        print("✅ No Azure credentials detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
