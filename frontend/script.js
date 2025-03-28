const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imageGrid = document.getElementById('imageGrid');
const output = document.getElementById('output');
const spinner = document.querySelector('.spinner');
const downloadBtn = document.getElementById('downloadJson');

let detectionResults = []; // Store results for multiple images

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.style.borderColor = "#6610f2";
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = "#007bff";
});

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    fileInput.files = event.dataTransfer.files;
    displayImages(fileInput.files);
});

fileInput.addEventListener('change', function () {
    if (fileInput.files.length) {
        displayImages(fileInput.files);
    }
});

function displayImages(files) {
    imageGrid.innerHTML = "";
    detectionResults = []; // Reset results

    Array.from(files).forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const wrapper = document.createElement("div");
            wrapper.classList.add("image-wrapper");

            const img = document.createElement("img");
            img.src = e.target.result;
            img.setAttribute("data-index", index);

            const removeBtn = document.createElement("button");
            removeBtn.classList.add("remove-btn");
            removeBtn.innerHTML = "❌";
            removeBtn.onclick = () => {
                wrapper.remove();
                removeImage(index);
            };

            wrapper.appendChild(img);
            wrapper.appendChild(removeBtn);
            imageGrid.appendChild(wrapper);
        };
        reader.readAsDataURL(file);
    });
}

function removeImage(index) {
    const updatedFiles = Array.from(fileInput.files).filter((_, i) => i !== index);
    const dataTransfer = new DataTransfer();
    updatedFiles.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
}

async function uploadImages() {
    if (!fileInput.files.length) {
        alert("Please upload at least one image.");
        return;
    }

    spinner.style.display = 'block';
    output.innerHTML = "";
    detectionResults = [];

    for (const file of fileInput.files) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData,
            });

            const result = await response.json();
            detectionResults.push(result);

            output.innerHTML += `<p><strong>Predicted Emoji:</strong> ${result.emoji}</p>`;
        } catch (error) {
            output.innerHTML += `<p>❌ Error detecting emoji for ${file.name}!</p>`;
        }
    }

    spinner.style.display = 'none';
    output.style.display = 'block';
    if (detectionResults.length) {
        downloadBtn.style.display = 'block';
    }
}

function downloadJson() {
    const blob = new Blob([JSON.stringify(detectionResults, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "emoji_detection_results.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
