from pathlib import Path

# CURRENT PROJECT DIRECTORY
# "." means the folder where this script is being run.
project_path = Path(".")

# FOLDERS
folders = [
    "graph",
    "scripts",
    "data",
]


# FILES
files = [
    "graph/state.py",
    
    "README.md",
    "requirements.txt",
    ".env.example",
    ".env",
]

# CREATE FOLDERS
for folder in folders:
    folder_path = project_path / folder
    folder_path.mkdir(
        exist_ok=True
    )

# CREATE FILES
for file in files:
    file_path = project_path / file
    file_path.touch(
        exist_ok=True
    )


# SUCCESS MESSAGE
print("\nProject structure created successfully!\n")
print("Project location:")
print(project_path.resolve())