# Medical-Chatbot

## Run

### Steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/heidiie13/medical-chatbot.git
    ```

2. **Create `.env` file:**
    ```sh
    cp .env.example .env
    ```

3. **Create and activate a conda environment:**
    ```sh
    conda create -n medical python=3.11.9 -y
    conda activate medical
    ```

4. **Install the requirements:**
    ```sh
    pip install -r requirements.txt
    ```

5. **Run Docker:**
    - Open Docker engine
    ```sh
    docker-compose up --build
    ```

6. **Seed data:**
    ```sh
    cd src
    python chunking.py
    python seed_data.py
    ```

7. **Run the UI:**
    ```sh
    streamlit run app.py
    ```
