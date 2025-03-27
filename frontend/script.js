const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imageGrid = document.getElementById('imageGrid');
const output = document.getElementById('output');
const spinner = document.querySelector('.spinner');
const downloadBtn = document.getElementById('downloadJson');
let detectionResult = null;

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

fileInput.addEventListener('change', function(event) {
    if (fileInput.files.length) {
        displayImages(fileInput.files);
    }
});

function displayImages(files) {
    imageGrid.innerHTML = "";

    Array.from(files).forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const wrapper = document.createElement("div");
            wrapper.classList.add("image-wrapper");

            const img = document.createElement("img");
            img.src = e.target.result;

            const removeBtn = document.createElement("button");
            removeBtn.classList.add("remove-btn");
            removeBtn.innerHTML = "❌";
            removeBtn.onclick = () => wrapper.remove();

            wrapper.appendChild(img);
            wrapper.appendChild(removeBtn);
            imageGrid.appendChild(wrapper);
        };
        reader.readAsDataURL(file);
    });
}

function uploadImages() {
    spinner.style.display = 'block';
    setTimeout(() => {
        spinner.style.display = 'none';
        output.style.display = 'block';
        output.innerHTML = '{ "status": "success", "message": "Emoji detected!" }';
        detectionResult = { status: "success", message: "Emoji detected!" };
        downloadBtn.style.display = 'block';
    }, 2000);
}

function downloadJson() {
    const blob = new Blob([JSON.stringify(detectionResult, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "emoji_detection_result.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
