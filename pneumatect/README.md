# PneumaTect Laravel Platform

> 🧩 Web Platform Component for the PneumaTect Project  
> 🔗 Built using Laravel + Jetstream + Blade + Falcon Admin Template  
> 🧠 Purpose: UI & API bridge for AI-based lung cancer diagnostic services

---

## 🚀 Overview

This repository hosts the **Laravel web platform** for **PneumaTect**, an AI-powered system for detecting and classifying pulmonary nodules. The Laravel frontend interfaces with the AI inference backend via RESTful APIs and provides a clean UI for clinicians, researchers, and healthcare staff.

---

## 🔧 Features

- Jetstream authentication with optional team support
- Modular admin panel based on Falcon Bootstrap v2.8.2
- Dashboard with dynamic charts, logs, and scan history
- Upload and visualize DICOM/CT images
- Trigger AI inference via Python-based API
- View diagnostic results, reports, and PDF exports
- Role-based access control and audit logging

---

## 🛠️ Tech Stack

| Layer         | Tech Used                     |
|---------------|-------------------------------|
| Backend       | Laravel 11 (PHP 8.3+)         |
| Frontend      | Blade Templates, TailwindCSS  |
| Auth & UI     | Jetstream + Livewire          |
| Admin UI      | Falcon Bootstrap v2.8.2       |
| Assets        | Gulp, Laravel Mix             |
| API Layer     | Axios (HTTP client)           |
| Integration   | Python Flask FastAPI backend  |
| Database      | MySQL                         |

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- PHP 8.3+
- Composer
- Node.js and npm
- MySQL (or another compatible database configured in Laravel)

---

## 🧪 Development Setup

Follow these steps to get your development environment set up:

1.  **Clone the repository:**


2.  **Install PHP dependencies:**
    ```bash
    composer install
    ```

3.  **Install Node dependencies and build assets:**
    ```bash
    npm install && npm run dev
    ```

4.  **Configure your environment:**
    Copy the example environment file and generate an application key.
    ```bash
    cp .env.example .env
    php artisan key:generate
    ```
    Then, edit the `.env` file to set your database credentials, API endpoint, and other necessary configurations.

5.  **Run database migrations and seeders:**
    ```bash
    php artisan migrate --seed
    ```

6.  **Serve the application:**
    ```bash
    php artisan serve
    ```
    The application should now be running on `http://localhost:8000` (or your configured port).

---

## 🔗 API Integration

All predictions and scan analysis are delegated to a separate Python AI Inference API. This Laravel platform sends HTTP requests to this API and visualizes the results.

### Example Flow

1.  User uploads a CT scan through the Laravel UI.
2.  Laravel backend sends a POST request to the `/predict/scan` endpoint of the Python AI Inference API.
3.  The AI API processes the scan and returns classification results, heatmap URLs, or other relevant diagnostic data.
4.  The Laravel UI renders these predictions, reports, and diagnostic overlays for the user.

### Configuration

Configure the necessary API endpoints and keys in your `.env` file


📁 Folder Structure
Here's an overview of the key directories in the project:

├── app/
│   ├── Http/Controllers/   # Application controllers
│   ├── Models/             # Eloquent models
│   └── Services/           # Services, including API clients
├── config/                 # Application configuration files
├── public/
│   └── assets/             # Compiled assets, Falcon template assets
├── resources/
│   ├── css/                # CSS source files
│   ├── js/                 # JavaScript source files
│   ├── views/              # Blade templates
│   └── components/         # Blade components
├── routes/
│   └── web.php             # Web routes
├── .env                    # Environment configuration (local)
├── composer.json           # PHP dependencies
└── package.json            # Node.js dependencies


---

## ✨ Developer Credits

👑 **Lead Developers**

| Name                               | Role                                                                                               |
|------------------------------------|----------------------------------------------------------------------------------------------------|
| Mohamed Amine Ghorbali (aka Aethelios) | Project Lead – Platform Engineering|
| Rouaa Guesmi                       | Full Stack Developer – Data Scientist|

*Your relentless effort, precision, and teamwork were the backbone of this rapid and clean platform deployment. Bravo 👏*

---

## 📜 License

This Laravel platform is part of the PneumaTect academic project at ESPRIT University.
It is licensed under the **MIT License** for non-commercial academic use. Please see the `LICENSE` file for more details.

---

## 📬 Contact & Contribution

We welcome contributions to improve the PneumaTect platform or adapt it for other medical use cases!

-   **Report issues or suggest features:** Use the GitHub Issues tab.
-   **Contribute code:** Fork the repository and submit a Pull Request.
-   **General inquiries:** Reach out to the PneumaTect team at `contact@pneumatect.ai`.
-   **Repository:** [https://github.com/rouaaguesmi1/PulmoScanProject/](https://github.com/rouaaguesmi1/PulmoScanProject/)

---

🫁 **PneumaTect: Breathing intelligence into lung health diagnostics.**
