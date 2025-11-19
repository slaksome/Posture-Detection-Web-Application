const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const video = document.getElementById("video");
const videoContainer = document.getElementById("video-container");
const statsDiv = document.getElementById("stats");

startBtn.onclick = async () => {
  statsDiv.textContent = "Starting camera...";
  try {
    const res = await fetch("/start", { method: "POST" });
    const data = await res.json();
    if (data.status === "started") {
      video.src = "/video_feed";
      videoContainer.style.display = "block";
      statsDiv.textContent = "";
    } else {
      statsDiv.textContent = "Error: " + (data.message || "Could not start");
    }
  } catch (err) {
    statsDiv.textContent = "Error starting: " + err;
  }
};

stopBtn.onclick = async () => {
  statsDiv.textContent = "Stopping...";
  try {
    const res = await fetch("/stop", { method: "POST" });
    const data = await res.json();

    const text =
      `Session Stopped\n\n` +
      `Good Frames: ${data.good_frames}\n` +
      `Bad Frames: ${data.bad_frames}\n\n` +
      `Good Posture: ${data.good_pct}%\n` +
      `Bad Posture: ${data.bad_pct}%`;

    statsDiv.textContent = text;

    video.src = "";
    videoContainer.style.display = "none";
  } catch (err) {
    statsDiv.textContent = "Error stopping: " + err;
  }
};
