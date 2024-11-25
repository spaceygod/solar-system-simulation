# GitHub Basics for New Users (Using VS Code)

This guide will help you quickly learn how to **pull** and **push** changes to a GitHub repository using **VS Code**.

---

## **1. Clone the Repository**

To start working on a repository, you need to clone it:

1. Open VS Code.
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open the Command Palette.
3. Search for **"Git: Clone"**.
4. Paste the repository URL and select a folder on your computer.
5. The repository will open in VS Code.

---

## **2. Pull Before You Push**

Always pull the latest changes before making or pushing your updates. This ensures your work is up-to-date with the team.

1. Click on the **Source Control** icon in the VS Code sidebar.
2. Click **Pull** (down arrow button) from the toolbar.

Alternatively, use the terminal:

```bash
git pull origin main
```

### **Why is this important?**

If you don’t pull first, your changes might conflict with updates made by others.

---

## **3. Make Changes Locally**

1. Edit files as needed in VS Code.
2. Save your changes.

---

## **4. Stage and Commit Your Changes**

1. Go to the **Source Control** tab in VS Code.
2. You will see a list of changed files.
3. Click the **+** icon next to each file to stage it.
4. Write a commit message in the text box and click the **✔️ Commit** button.

Alternatively, use the terminal:

```bash
git add .
git commit -m "Your commit message"
```

---

## **5. Push Your Changes**

1. After committing, click **Push** (up arrow button) in the VS Code Source Control tab.

Alternatively, use the terminal:

```bash
git push origin main
```

---

## **Quick Workflow Summary**

1. **Pull**: Always pull the latest changes first.
2. **Make Changes**: Edit and save files.
3. **Stage & Commit**: Stage your changes and write a commit message.
4. **Push**: Push your changes to GitHub.

---

Of als alternative gebruik de sync optie:

1. **Pull**:
   Downloads the latest changes from the remote repository to your local repository.
   Purpose: To ensure your local copy is up-to-date with the latest changes made by others.

2. **Push**:
   Uploads your committed changes from your local repository to the remote repository.
   Purpose: To share your updates with others.

3. **Sync (in VS Code)**:
   Combines pull and push in one action.
   Purpose: To ensure your local and remote repositories are fully synchronized. It:
   Pulls the latest changes from the remote.
   Pushes your local changes to the remote.
   Simplified Explanation

- Pulling is like downloading the latest version of the project from GitHub.
- Pushing is like uploading your changes to GitHub.
- Syncing does both actions together (pull first, then push).

**Tip:** If you encounter conflicts while pulling, VS Code will highlight them in the editor. Resolve these conflicts before proceeding.

# Project Principles and Checklist

## General Principles

1. **Use comments:** Ensure your code is well-documented with meaningful comments to enhance readability.
2. **Write well-structured code:** Use consistent indentation and clear, logical organization.
3. **Use variables effectively:** Name variables meaningfully and avoid hardcoding values.
4. **Organize the project:**
   - Use separate directories for different content:
     - `utils/` for utility scripts.
     - `images/` for project-related images.
     - `tests/` for test cases.
     - `docs/` for additional documentation.

---

## Team Checklist

Track tasks and progress for each member below. Use checkboxes to mark completed tasks.

### **Uzay**

- [x] Do this
- [ ] Do that

### **Harsh**

- [x] idk
- [ ] n-body met numpy

### **Tijmen**

- [ ] Pending task 1
- [ ] Pending task blabla

### **Max**

- [x] Research topic
- [ ] blabla

---

## How to Use the Checklist

1. Add tasks under each team member's name.
2. Use `[x]` to mark tasks as completed and `[ ]` for incomplete tasks.
3. Update regularly to reflect progress.

This way we can track what everyone has done and what is being done.

---

## Example Directory Structure

project/ \
│ \
├── utils/ \
├── images/ \
├── tests/ \
├── docs/ \
├── README.md \
└── main.py

---

**Happy Coding!**
