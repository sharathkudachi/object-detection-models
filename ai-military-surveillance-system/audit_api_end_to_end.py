import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_api():
    print("\n--- AUDIT 7: Live API End-to-End Verification ---")

    # 1. System Info
    req = urllib.request.urlopen(f"{BASE_URL}/api/v1/system")
    sys_info = json.loads(req.read().decode())
    print(f"1. /api/v1/system -> Device: {sys_info['device']}, Models: {len(sys_info['models'])}, Profiles: {sys_info['profiles']}")
    assert "device" in sys_info

    # 2. Upload video
    video_path = Path("test_video.mp4")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    data = []
    data.append(f"--{boundary}".encode())
    data.append(f'Content-Disposition: form-data; name="file"; filename="{video_path.name}"'.encode())
    data.append(b"Content-Type: video/mp4")
    data.append(b"")
    data.append(video_path.read_bytes())
    data.append(f"--{boundary}--".encode())
    data.append(b"")
    body = b"\r\n".join(data)

    up_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    up_res = json.loads(urllib.request.urlopen(up_req).read().decode())
    uid = up_res["upload_id"]
    print(f"2. /api/v1/upload -> Upload ID: {uid}, Frames: {up_res['frames']}, Resolution: {up_res['width']}x{up_res['height']}")
    assert uid

    # 3. Start Analysis (Balanced Profile)
    start_payload = json.dumps({
        "upload_id": uid,
        "model": "best1.pt",
        "mode": "military+optical_flow",
        "profile": "fast",
        "stride": 3,
        "imgsz": 480
    }).encode("utf-8")

    start_req = urllib.request.Request(
        f"{BASE_URL}/api/v1/analysis/start",
        data=start_payload,
        headers={"Content-Type": "application/json"}
    )
    start_res = json.loads(urllib.request.urlopen(start_req).read().decode())
    aid = start_res["id"]
    print(f"3. /api/v1/analysis/start -> Job ID: {aid}, Initial Status: {start_res['status']}")

    # 4. Progress Reporting Poll
    print("4. Polling /api/v1/analysis/{aid} for live telemetry...")
    status = start_res["status"]
    poll_records = []
    t0 = time.time()
    while status not in ("completed", "failed", "cancelled"):
        time.sleep(0.8)
        poll_res = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/v1/analysis/{aid}").read().decode())
        status = poll_res["status"]
        poll_records.append((poll_res.get("stage"), poll_res.get("progress"), poll_res.get("fps"), poll_res.get("eta_s")))
        if time.time() - t0 > 60:
            raise TimeoutError("Job took too long to complete!")

    print(f"   Completed in {time.time()-t0:.1f}s. Final Status: {status}")
    print(f"   Sample Progress Polls: {poll_records[:4]} ... {poll_records[-1:]}")
    assert status == "completed", f"Job failed with status {status}"

    # 5. Playback Streaming (/api/v1/analysis/{aid}/video)
    vid_req = urllib.request.Request(f"{BASE_URL}/api/v1/analysis/{aid}/video")
    vid_res = urllib.request.urlopen(vid_req)
    content_type = vid_res.headers.get("Content-Type")
    vid_bytes = vid_res.read(1024)
    print(f"5. /api/v1/analysis/{aid}/video -> Content-Type: {content_type}, First chunk: {len(vid_bytes)} bytes")
    assert content_type == "video/mp4"
    assert len(vid_bytes) == 1024

    # 6. Video Download (/api/v1/analysis/{aid}/download)
    dl_req = urllib.request.Request(f"{BASE_URL}/api/v1/analysis/{aid}/download")
    dl_res = urllib.request.urlopen(dl_req)
    dl_type = dl_res.headers.get("Content-Type")
    dl_disposition = dl_res.headers.get("Content-Disposition")
    print(f"6. /api/v1/analysis/{aid}/download -> Content-Type: {dl_type}, Disposition: {dl_disposition}")
    assert "attachment" in dl_disposition or "annotated" in dl_disposition

    # 7. Result JSON (/api/v1/analysis/{aid}/result)
    res_req = urllib.request.Request(f"{BASE_URL}/api/v1/analysis/{aid}/result")
    res_json = json.loads(urllib.request.urlopen(res_req).read().decode())
    print(f"7. /api/v1/analysis/{aid}/result -> Frames: {len(res_json['frames'])}, Convoy Groups: {len(res_json['groups'])}, Avg FPS: {res_json['perf']['avg_fps']}")
    assert "meta" in res_json and "frames" in res_json and "perf" in res_json

    # 8. Test Cancellation Workflow
    print("8. Testing Cancellation Workflow...")
    start_req2 = urllib.request.Request(
        f"{BASE_URL}/api/v1/analysis/start",
        data=start_payload,
        headers={"Content-Type": "application/json"}
    )
    job2 = json.loads(urllib.request.urlopen(start_req2).read().decode())
    aid2 = job2["id"]
    time.sleep(0.5)

    stop_req = urllib.request.Request(f"{BASE_URL}/api/v1/analysis/{aid2}/stop", data=b"{}", headers={"Content-Type": "application/json"})
    stop_res = json.loads(urllib.request.urlopen(stop_req).read().decode())
    print(f"   Cancellation Signal Sent: {stop_res}")

    # Wait for cancellation to settle
    time.sleep(1.0)
    job2_status = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/v1/analysis/{aid2}").read().decode())
    print(f"   Job 2 Status after Stop: {job2_status['status']} (Stage: {job2_status['stage']})")
    assert job2_status["status"] == "cancelled" or job2_status.get("cancel") is True

    # 9. Test Storage Cleanup
    clean_req = urllib.request.Request(f"{BASE_URL}/api/v1/jobs/cleanup?max_age_hours=0", data=b"{}", headers={"Content-Type": "application/json"})
    clean_res = json.loads(urllib.request.urlopen(clean_req).read().decode())
    print(f"9. /api/v1/jobs/cleanup -> Cleaned {clean_res.get('cleaned_files')} old files")

    print("\nALL END-TO-END API TESTS PASSED!")

if __name__ == "__main__":
    test_api()
