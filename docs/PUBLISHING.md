# Publishing Guide

This repository is prepared for GitHub and Comfy Registry publication.

## 1. Before Publishing

Check these values in `pyproject.toml`:

- `[project].name`
- `[project].version`
- `[project.urls].Repository`
- `[tool.comfy].PublisherId`
- `[tool.comfy].DisplayName`

The current placeholder repository URL is:

```text
https://github.com/peter119lee/ComfyUI-RUM
```

Change it if publishing under another GitHub account or organization.

## License Warning

Do not make the repository public or publish to Comfy Registry until the upstream RUM license/permission question is resolved. This wrapper mirrors parts of the upstream inference implementation.

## 2. Registry Account

Create a publisher on Comfy Registry and make sure `PublisherId` in `pyproject.toml` matches that publisher id.

## 3. Manual Publish

Install Comfy CLI and run:

```bash
comfy node publish
```

## 4. GitHub Actions Publish

The repository includes:

```text
.github/workflows/publish_action.yml
```

To use it:

1. Go to GitHub repository settings.
2. Add repository secret `REGISTRY_ACCESS_TOKEN`.
3. Run the workflow manually from the GitHub Actions tab.

It is intentionally `workflow_dispatch` only, so a missing secret will not break every normal push.

## 5. Release Checklist

- `python scripts/check_install.py` passes in a ComfyUI Python environment.
- `README.md` has accurate install and model download instructions.
- `CHANGELOG.md` has the new version.
- `pyproject.toml` version is bumped.
- No files under `models/` are committed.
- No `local_config.json` is committed.
