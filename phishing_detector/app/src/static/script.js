
document.addEventListener("DOMContentLoaded", function() {
    // Smooth scrolling for navigation links
    document.querySelectorAll("a.nav-link").forEach(anchor => {
        anchor.addEventListener("click", function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute("href")).scrollIntoView({
                behavior: "smooth"
            });
        });
    });

    // URL Checker
    document.getElementById("checkUrlBtn").addEventListener("click", async function() {
        const urlInput = document.getElementById("urlInput").value;
        const urlResultDiv = document.getElementById("urlResult");
        urlResultDiv.innerHTML = 
            `<div class="alert alert-info">Checking URL...</div>`;

        try {
            const response = await fetch("/predict_url", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ url: urlInput })
            });
            const data = await response.json();

            if (response.ok) {
                const label = data.label;
                const score = (data.score * 100).toFixed(2);
                const badgeClass = label === "phishing" ? "badge-phishing" : "badge-legitimate";
                const displayLabel = label.charAt(0).toUpperCase() + label.slice(1);
                urlResultDiv.innerHTML = 
                    `<div class="alert alert-success">Result: <span class="badge ${badgeClass}">${displayLabel}</span> with ${score}% confidence.</div>`;
            } else {
                urlResultDiv.innerHTML = 
                    `<div class="alert alert-danger">Error: ${data.error || "Something went wrong"}</div>`;
            }
        } catch (error) {
            console.error("Error checking URL:", error);
            urlResultDiv.innerHTML = 
                `<div class="alert alert-danger">An error occurred while connecting to the server.</div>`;
        }
    });

    // Email Checker
    document.getElementById("checkEmailBtn").addEventListener("click", async function() {
        const emailSubject = document.getElementById("emailSubject").value;
        const emailBody = document.getElementById("emailBody").value;
        const emailResultDiv = document.getElementById("emailResult");
        emailResultDiv.innerHTML = 
            `<div class="alert alert-info">Checking Email...</div>`;

        try {
            const response = await fetch("/predict_email", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ subject: emailSubject, body: emailBody })
            });
            const data = await response.json();

            if (response.ok) {
                const label = data.label;
                const score = (data.score * 100).toFixed(2);
                const badgeClass = label === "phishing" ? "badge-phishing" : "badge-not-phishing";
                const displayLabel = label.charAt(0).toUpperCase() + label.slice(1).replace("_", " ");
                emailResultDiv.innerHTML = 
                    `<div class="alert alert-success">Result: <span class="badge ${badgeClass}">${displayLabel}</span> with ${score}% confidence.</div>`;
            } else {
                emailResultDiv.innerHTML = 
                    `<div class="alert alert-danger">Error: ${data.error || "Something went wrong"}</div>`;
            }
        } catch (error) {
            console.error("Error checking email:", error);
            emailResultDiv.innerHTML = 
                `<div class="alert alert-danger">An error occurred while connecting to the server.</div>`;
        }
    });
});


