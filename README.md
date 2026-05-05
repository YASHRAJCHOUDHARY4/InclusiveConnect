# InclusiveConnect Backend

Welcome to the **InclusiveConnect-Backend** repository. This is a Python-based backend application designed to handle the core server-side logic, routing, and data processing for the InclusiveConnect platform. 

The project is built to be easily deployable on Replit and uses standard Python templating for its web interfaces.

## 📁 Project Structure

Here is an overview of the core files and directories in this repository:

* **`main.py`**: The primary entry point for the backend application.
* **`pyproject.toml` & `poetry.lock`**: Configuration files used by Poetry to manage project dependencies and virtual environments[cite: 1].
* **`templates/`**: Directory containing web templates, specifically the core `index.html` file used to render the web interface[cite: 1].
* **`.replit`**: Configuration file that allows this project to be instantly run and hosted within the Replit environment[cite: 1].
* **`.gitignore`**: Specifies intentionally untracked files that Git should ignore[cite: 1].
* **`__pycache__/`**: System-generated directory containing compiled Python bytecode (`.pyc` files) for Python 3.10 and 3.11, which helps speed up module loading[cite: 1].
* **`.breakpoints` & `.local/`**: Internal development and environment configuration files[cite: 1].

## 🛠️ Tech Stack

* **Language**: Python (3.10 / 3.11)
* **Dependency Management**: [Poetry](https://python-poetry.org/)
* **Environment**: [Replit](https://replit.com/) (Native Support)
* **Frontend/Views**: HTML5 (via Python templating)

## 🚀 Getting Started

### Prerequisites

If you are running this locally (outside of Replit), ensure you have the following installed:
* Python 3.10 or higher
* Poetry

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd InclusiveConnect-Backend
2. Install dependencies:
  ```bash
    poetry install
  ```
3. **Run the application:**
   Start the backend server using:
   ```bash
   poetry run python main.py


## Running on Replit
Since this project includes a .replit configuration file, you can easily fork and run it directly in your browser.

1. Import the repository into a new Replit workspace.

2. Click the large Run button at the top of the interface. Replit will automatically read the .replit file, install the Poetry dependencies, and execute main.py

## 🤝 Contributing
1. Fork the repository.

2. Create your feature branch (git checkout -b feature/AmazingFeature).

3. Commit your changes (git commit -m 'Add some AmazingFeature').

4. Push to the branch (git push origin feature/AmazingFeature).

5. Open a Pull Request.
