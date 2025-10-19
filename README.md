# AI Job Portal

An intelligent job portal built with Python Flask that uses AI to match job seekers with suitable job opportunities based on their skills and experience.

## Features

- **User Registration & Profiles**: Create detailed profiles with skills, experience, and education
- **AI-Powered Job Recommendations**: Get personalized job suggestions based on skill matching
- **Job Browsing**: Search and filter through available job opportunities
- **Employer Dashboard**: Post jobs and manage listings
- **Skills-Based Matching**: Uses TF-IDF vectorization and cosine similarity for intelligent recommendations

## Technology Stack

- **Backend**: Python Flask
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **AI/ML**: scikit-learn, NLTK, NumPy
- **Authentication**: Flask sessions with password hashing

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## Usage

### For Job Seekers
1. Register and create a detailed profile with your skills and experience
2. Login to view AI-recommended jobs with match percentages
3. Browse all available job opportunities
4. Contact employers directly using provided contact information

### For Employers
1. Use "Employer Login" to access the employer dashboard
2. Post detailed job listings with requirements and descriptions
3. Manage your posted jobs and view statistics

## AI Recommendation System

The AI system works by:
1. Converting job descriptions and user skills into numerical vectors using TF-IDF
2. Using cosine similarity to find the best matches
3. Providing percentage-based compatibility scores
4. Ranking recommendations by relevance

## API Endpoints

- `GET /`: Home page
- `GET/POST /register`: User registration
- `GET/POST /login`: User login
- `GET/POST /employer_login`: Employer login
- `GET /dashboard`: Job seeker dashboard with recommendations
- `GET /employer_dashboard`: Employer dashboard
- `GET/POST /post_job`: Job posting form
- `GET /jobs`: Browse all jobs
- `GET /api/recommendations/<user_id>`: API for job recommendations

## Security Features

- Password hashing using Werkzeug
- Session management
- SQL injection protection via SQLAlchemy
- Input validation and sanitization

## License

This project is licensed under the MIT License.