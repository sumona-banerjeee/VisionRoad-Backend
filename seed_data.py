"""
Seed script to populate database with dummy data for testing hierarchical APIs.
Uses absolute NHAI-style kilometer chainages.

Usage:
    python seed_data.py
"""

import os
import requests

_PORT = os.environ.get("PORT", "8000")
BASE_URL = f"http://localhost:{_PORT}/api/v1"


def create_project(name, state, corridor_name, start_lat, start_lng, end_lat, end_lng):
    """Create a project via API"""
    response = requests.post(
        f"{BASE_URL}/projects",
        json={
            "name": name,
            "state": state,
            "corridor_name": corridor_name,
            "start_lat": start_lat,
            "start_lng": start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
        },
    )
    if response.status_code == 201:
        data = response.json()
        print(f"✓ Created project: {name} (ID: {data['id']})")
        return data
    else:
        print(f"✗ Failed to create project {name}: {response.text}")
        return None


def create_package(project_id, name, region, chainage_start_km=None, chainage_end_km=None):
    """Create a package via API"""
    response = requests.post(
        f"{BASE_URL}/packages",
        json={
            "project_id": project_id,
            "name": name,
            "region": region,
            "chainage_start_km": chainage_start_km,
            "chainage_end_km": chainage_end_km,
        },
    )
    if response.status_code == 201:
        data = response.json()
        km_info = f" [KM {chainage_start_km}–{chainage_end_km}]" if chainage_start_km is not None else ""
        print(f"  ✓ Created package: {name}{km_info} (ID: {data['id']})")
        return data
    else:
        print(f"  ✗ Failed to create package {name}: {response.text}")
        return None


def create_chainage(
    package_id,
    segment_name,
    chainage_start_km,
    chainage_end_km,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
    direction="UP",
):
    """Create a chainage via API"""
    response = requests.post(
        f"{BASE_URL}/chainages",
        json={
            "package_id": package_id,
            "segment_name": segment_name,
            "chainage_start_km": chainage_start_km,
            "chainage_end_km": chainage_end_km,
            "start_lat": start_lat,
            "start_lng": start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
            "direction": direction,
        },
    )
    if response.status_code == 201:
        data = response.json()
        print(f"    ✓ Created chainage: {segment_name} [{direction}] [KM {chainage_start_km}–{chainage_end_km}] (ID: {data['id']})")
        return data
    else:
        print(f"    ✗ Failed to create chainage {segment_name}: {response.text}")
        return None


def seed_kolkata_jaipur_highway():
    """Seed Kolkata-Jaipur Highway (NH-19) project"""
    print("\n" + "=" * 60)
    print("Seeding: Kolkata-Jaipur Highway (NH-19)")
    print("=" * 60)

    project = create_project(
        name="Kolkata-Jaipur Highway Development",
        state="West Bengal, Bihar, Uttar Pradesh, Rajasthan",
        corridor_name="National Highway 19 (NH-19)",
        start_lat=22.5726,
        start_lng=88.3639,
        end_lat=26.9124,
        end_lng=75.7873,
    )
    if not project:
        return

    # Package 1: Kolkata-Patna Section (absolute KM 0–220)
    package1 = create_package(
        project_id=project["id"],
        name="Kolkata-Patna Section",
        region="West Bengal - Bihar",
        chainage_start_km=0,
        chainage_end_km=220,
    )
    if package1:
        create_chainage(package1["id"], "Kolkata South Corridor", 0, 10,
                             22.569, 88.430, 22.582, 88.438, "UP")
        create_chainage(package1["id"], "Kolkata South Corridor", 0, 10,
                             22.569, 88.430, 22.582, 88.438, "DOWN")

        create_chainage(package1["id"], "Dankuni-Burdwan", 10, 80,
                             22.6500, 88.2800, 23.2500, 87.8600, "UP")
        create_chainage(package1["id"], "Dankuni-Burdwan", 10, 80,
                             22.6500, 88.2800, 23.2500, 87.8600, "DOWN")

        create_chainage(package1["id"], "Burdwan-Asansol", 80, 160,
                             23.2500, 87.8600, 23.6850, 86.9830, "UP")

        create_chainage(package1["id"], "Asansol-Dhanbad", 160, 220,
                             23.6850, 86.9830, 23.7957, 86.4304, "UP")

    # Package 2: Patna-Varanasi Section (absolute KM 400–600)
    package2 = create_package(
        project_id=project["id"],
        name="Patna-Varanasi Section",
        region="Bihar - Uttar Pradesh",
        chainage_start_km=400,
        chainage_end_km=600,
    )
    if package2:
        create_chainage(package2["id"], "Patna-Aurangabad", 400, 500,
                             25.5941, 85.1376, 24.7521, 84.3742, "UP")

        create_chainage(package2["id"], "Aurangabad-Varanasi", 500, 600,
                             24.7521, 84.3742, 25.3176, 82.9739, "UP")

    # Package 3: Varanasi-Agra Section (absolute KM 600–900)
    package3 = create_package(
        project_id=project["id"],
        name="Varanasi-Agra Section",
        region="Uttar Pradesh",
        chainage_start_km=600,
        chainage_end_km=900,
    )
    if package3:
        create_chainage(package3["id"], "Varanasi-Allahabad", 600, 720,
                             25.3176, 82.9739, 25.4358, 81.8463, "UP")

        create_chainage(package3["id"], "Allahabad-Kanpur", 720, 900,
                             25.4358, 81.8463, 26.4499, 80.3319, "DOWN")


def seed_mumbai_pune_expressway():
    """Seed Mumbai-Pune Expressway project"""
    print("\n" + "=" * 60)
    print("Seeding: Mumbai-Pune Expressway")
    print("=" * 60)

    project = create_project(
        name="Mumbai-Pune Expressway Maintenance",
        state="Maharashtra",
        corridor_name="Mumbai-Pune Expressway (NH-48)",
        start_lat=18.9220,
        start_lng=72.8347,
        end_lat=18.5204,
        end_lng=73.8567,
    )
    if not project:
        return

    # Package 1: Mumbai-Lonavala Section (KM 0–70)
    package1 = create_package(
        project_id=project["id"],
        name="Mumbai-Lonavala Section",
        region="Mumbai Metropolitan Region",
        chainage_start_km=0,
        chainage_end_km=70,
    )
    if package1:
        create_chainage(package1["id"], "Kalyan-Khopoli", 0, 40,
                             19.2403, 73.1305, 18.7854, 73.3457, "UP")

        create_chainage(package1["id"], "Khopoli-Lonavala", 40, 70,
                             18.7854, 73.3457, 18.7537, 73.4087, "UP")

    # Package 2: Lonavala-Pune Section (KM 70–95)
    package2 = create_package(
        project_id=project["id"],
        name="Lonavala-Pune Section",
        region="Pune District",
        chainage_start_km=70,
        chainage_end_km=95,
    )
    if package2:
        create_chainage(package2["id"], "Lonavala-Talegaon", 70, 85,
                             18.7537, 73.4087, 18.7351, 73.6765, "DOWN")

        create_chainage(package2["id"], "Talegaon-Pune", 85, 95,
                             18.7351, 73.6765, 18.5204, 73.8567, "DOWN")


def seed_delhi_chandigarh_highway():
    """Seed Delhi-Chandigarh Highway project"""
    print("\n" + "=" * 60)
    print("Seeding: Delhi-Chandigarh Highway (NH-44)")
    print("=" * 60)

    project = create_project(
        name="Delhi-Chandigarh Highway Upgrade",
        state="Delhi, Haryana, Punjab",
        corridor_name="National Highway 44 (NH-44)",
        start_lat=28.7041,
        start_lng=77.1025,
        end_lat=30.7333,
        end_lng=76.7794,
    )
    if not project:
        return

    # Package 1: Delhi-Panipat Section (KM 0–90)
    package1 = create_package(
        project_id=project["id"],
        name="Delhi-Panipat Section",
        region="Delhi-Haryana Border",
        chainage_start_km=0,
        chainage_end_km=90,
    )
    if package1:
        create_chainage(package1["id"], "Delhi-Sonipat", 0, 45,
                             28.7041, 77.1025, 28.9931, 77.0151, "UP")

        create_chainage(package1["id"], "Sonipat-Panipat", 45, 90,
                             28.9931, 77.0151, 29.3909, 76.9635, "UP")

    # Package 2: Panipat-Chandigarh Section (KM 90–270)
    package2 = create_package(
        project_id=project["id"],
        name="Panipat-Chandigarh Section",
        region="Haryana-Punjab Border",
        chainage_start_km=90,
        chainage_end_km=270,
    )
    if package2:
        create_chainage(package2["id"], "Panipat-Karnal", 90, 150,
                             29.3909, 76.9635, 29.6857, 76.9905, "UP")

        create_chainage(package2["id"], "Karnal-Ambala", 150, 220,
                             29.6857, 76.9905, 30.3782, 76.7821, "DOWN")

        create_chainage(package2["id"], "Ambala-Chandigarh", 220, 270,
                             30.3782, 76.7821, 30.7333, 76.7794, "UP")


def main():
    """Run all seed functions"""
    print("\n" + "=" * 60)
    print("     VisionRoad Database Seeding Script")
    print("=" * 60)
    print(f"Target API: {BASE_URL}")
    print("=" * 60)

    try:
        response = requests.get(f"http://localhost:{_PORT}/")
        print("✓ API server is running\n")
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to API server!")
        print("  Make sure the server is running: uvicorn main:app --reload")
        return

    seed_kolkata_jaipur_highway()
    seed_mumbai_pune_expressway()
    seed_delhi_chandigarh_highway()

    print("\n" + "=" * 60)
    print("✓ Seeding complete!")
    print("=" * 60)
    print("\nTest the APIs:")
    print(f"  - List projects:   GET {BASE_URL}/projects")
    print(f"  - List packages:   GET {BASE_URL}/packages")
    print(f"  - List chainages:  GET {BASE_URL}/chainages")
    print(f"  - Project summary: GET {BASE_URL}/summary/projects/<project_id>")
    print(f"  - API docs:        http://localhost:{_PORT}/docs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
