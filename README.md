# Interactive Dashboard for Social Platform Exploration

The aim of the project will be to create an interactive dashboard to analyse various social media datasets using the Streamlit tool.

### Authors

- [Judyta Bąkowska](https://github.com/JudytaBakowska)
- [Bianka Marciniak](https://github.com/bmarciniak)
- [Rafał Tekielski](https://github.com/Rafal354)

### How to run the project with Docker

1. Clone the repository

```bash
git clone https://github.com/JudytaBakowska/Exploring_Social_Platforms.git
```
2. Go to the project directory

```bash
cd Exploring_Social_Platforms
```

3. Build the Docker image

```bash
docker build -t exploring_social_platforms .
```

4. Run the Docker container

```bash
docker run -p 8501:8501 exploring_social_platforms
```

### How to run the project locally

1. Clone the repository

```bash
git clone https://github.com/JudytaBakowska/Exploring_Social_Platforms.git
```

2. Go to the project directory

```bash
cd Exploring_Social_Platforms
```

3. Install the required packages

```bash
pip install -r requirements.txt 
```

4. Run the Streamlit app

```bash
streamlit run app.py
```