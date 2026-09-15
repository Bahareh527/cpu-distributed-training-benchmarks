# Contributing

Contributions are welcome through an issue or pull request.

Before submitting a change:

1. Run `ruff check .`.
2. Run `pytest`.
3. Run the two-worker synthetic smoke commands for parameter server, ring, and DDP.
4. Do not commit downloaded datasets, generated logs, credentials, or machine-specific paths.
5. Describe hardware and run settings with any benchmark result.
