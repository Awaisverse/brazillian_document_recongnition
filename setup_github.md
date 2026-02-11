# GitHub Setup

Git is already initialized. Next steps:

## Configure Git (if not done before)

```bash
git config --global user.email "your@email.com"
git config --global user.name "Your Name"
```

## Commit and push

1. Add files and commit:

```bash
git add .
git commit -m "Initial commit: Brazilian document classification (60/40 split)"
```

## Create and push to GitHub

1. Create a new repository on [GitHub](https://github.com/new):
   - Name: `brazilian-document-recognition` (or similar)
   - Do **not** initialize with README (you already have one)

2. Add remote and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/brazilian-document-recognition.git
git branch -M main
git push -u origin main
```

3. If using SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/brazilian-document-recognition.git
git push -u origin main
```

## Note

This project lives inside a parent repo (`github files`). For a clean GitHub repo, run `git init` and the commands above **inside** `brazillian_document_recongnition` so it becomes its own independent repository.
