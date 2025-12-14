# Celestial Linking - Quantum Physics Problem Solver

## Overview
Celestial Linking is a full-stack web application that helps users solve quantum physics problems. The platform provides a library of quantum equations, interactive problem solvers, visualizations, and export capabilities.

## Tech Stack
- **Backend**: Python Flask
- **Database**: PostgreSQL (via Replit's built-in database)
- **Authentication**: Replit Auth (OpenID Connect)
- **Visualization**: Matplotlib
- **Data Export**: Pandas + openpyxl
- **Frontend**: HTML/CSS/JavaScript with Bootstrap 5

## Project Structure
```
/
├── app.py              # Flask app initialization and database setup
├── main.py             # Application entry point
├── models.py           # SQLAlchemy database models
├── routes.py           # Flask routes and API endpoints
├── replit_auth.py      # Replit Auth integration
├── quantum_solver.py   # Quantum physics calculations and graph generation
├── templates/          # HTML templates
│   ├── base.html       # Base template with navigation
│   ├── index.html      # Landing page
│   ├── library.html    # Quantum equation library
│   ├── solver.html     # Problem solver interface
│   ├── problems.html   # Saved problems view
│   ├── runs.html       # Session run history
│   ├── account.html    # User account page
│   ├── 403.html        # Access denied page
│   └── 404.html        # Not found page
├── static/
│   ├── css/style.css   # Custom styles
│   └── js/main.js      # Frontend JavaScript
└── replit.md           # Project documentation
```

## Features

### 1. Quantum Library
Searchable library of quantum equations:
- Particle in a Box (Infinite Square Well)
- Quantum Harmonic Oscillator
- Hydrogen Atom Radial Function
- Quantum Tunneling
- Free Particle Wave Packet

### 2. Problem Solver
- Input custom parameters for each equation
- Step-by-step solution explanations
- Numerical results with scientific notation
- 1D graph visualizations (wavefunction + probability density)

### 3. User Accounts
- Secure authentication via Replit Auth
- Persistent storage of solved problems
- Problem history automatically loads on login

### 4. Export Functionality
- Download graphs as JPG images
- Export solutions as Excel spreadsheets

## Database Models

### User
- Stores user profile from Replit Auth
- Links to solved problems

### SolvedProblem
- equation_type: Type of equation solved
- equation_name: Display name
- parameters: JSON-encoded input parameters
- solution: JSON-encoded results
- steps: JSON-encoded step-by-step explanation
- graph_data: JSON-encoded plot data

## Running the Application
The app runs on port 5000 with Flask's development server:
```bash
python main.py
```

For production:
```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port main:app
```

## Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (auto-configured)
- `SESSION_SECRET`: Flask session secret key
- `REPL_ID`: Replit ID for authentication
