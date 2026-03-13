"""
Seed script to populate database with dummy data for testing hierarchical APIs.

Usage:
    python seed_data.py
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


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


def create_package(project_id, name, region):
    """Create a package via API"""
    response = requests.post(
        f"{BASE_URL}/packages",
        json={"project_id": project_id, "name": name, "region": region},
    )
    if response.status_code == 201:
        data = response.json()
        print(f"  ✓ Created package: {name} (ID: {data['id']})")
        return data
    else:
        print(f"  ✗ Failed to create package {name}: {response.text}")
        return None


def create_location(
    package_id,
    segment_name,
    chainage_start,
    chainage_end,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
):
    """Create a location via API"""
    response = requests.post(
        f"{BASE_URL}/locations",
        json={
            "package_id": package_id,
            "segment_name": segment_name,
            "chainage_start_km": chainage_start,
            "chainage_end_km": chainage_end,
            "start_lat": start_lat,
            "start_lng": start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
        },
    )
    if response.status_code == 201:
        data = response.json()
        print(f"    ✓ Created location: {segment_name} (ID: {data['id']})")
        return data
    else:
        print(f"    ✗ Failed to create location {segment_name}: {response.text}")
        return None


def seed_kolkata_jaipur_highway():
    """Seed Kolkata-Jaipur Highway (NH-19) project"""
    print("\n" + "=" * 60)
    print("Seeding: Kolkata-Jaipur Highway (NH-19)")
    print("=" * 60)

    # Create project
    project = create_project(
        name="Kolkata-Jaipur Highway Development",
        state="West Bengal, Bihar, Uttar Pradesh, Rajasthan",
        corridor_name="National Highway 19 (NH-19)",
        start_lat=22.5726,  # Kolkata
        start_lng=88.3639,
        end_lat=26.9124,  # Jaipur
        end_lng=75.7873,
    )

    if not project:
        return

    # Package 1: Kolkata-Patna Section
    package1 = create_package(
        project_id=project["id"],
        name="Kolkata-Patna Section",
        region="West Bengal - Bihar",
    )

    if package1:
        # Location 0
        create_location(
            package_id=package1["id"],
            segment_name="Kolkata South Corridor",
            start_lat=22.569,
            start_lng=88.430,
            end_lat=22.582,
            end_lng=88.438,
            chainage_start=0,
            chainage_end=10,
        )
        # Location 1: Dankuni-Burdwan segment
        create_location(
            package_id=package1["id"],
            segment_name="Dankuni-Burdwan (KM 0-80)",
            chainage_start=10,
            chainage_end=80.0,
            start_lat=22.6500,  # Dankuni area
            start_lng=88.2800,
            end_lat=23.2500,  # Burdwan area
            end_lng=87.8600,
        )

        # Location 2: Burdwan-Asansol segment
        create_location(
            package_id=package1["id"],
            segment_name="Burdwan-Asansol (KM 80-160)",
            chainage_start=80.0,
            chainage_end=160.0,
            start_lat=23.2500,  # Burdwan
            start_lng=87.8600,
            end_lat=23.6850,  # Asansol
            end_lng=86.9830,
        )

        # Location 3: Asansol-Dhanbad segment
        create_location(
            package_id=package1["id"],
            segment_name="Asansol-Dhanbad (KM 160-220)",
            chainage_start=160.0,
            chainage_end=220.0,
            start_lat=23.6850,  # Asansol
            start_lng=86.9830,
            end_lat=23.7957,  # Dhanbad
            end_lng=86.4304,
        )

    # Package 2: Patna-Varanasi Section
    package2 = create_package(
        project_id=project["id"],
        name="Patna-Varanasi Section",
        region="Bihar - Uttar Pradesh",
    )

    if package2:
        # Location 1: Patna-Aurangabad segment
        create_location(
            package_id=package2["id"],
            segment_name="Patna-Aurangabad (KM 0-100)",
            chainage_start=0.0,
            chainage_end=100.0,
            start_lat=25.5941,  # Patna
            start_lng=85.1376,
            end_lat=24.7521,  # Aurangabad
            end_lng=84.3742,
        )

        # Location 2: Aurangabad-Varanasi segment
        create_location(
            package_id=package2["id"],
            segment_name="Aurangabad-Varanasi (KM 100-200)",
            chainage_start=100.0,
            chainage_end=200.0,
            start_lat=24.7521,  # Aurangabad
            start_lng=84.3742,
            end_lat=25.3176,  # Varanasi
            end_lng=82.9739,
        )

    # Package 3: Varanasi-Agra Section
    package3 = create_package(
        project_id=project["id"], name="Varanasi-Agra Section", region="Uttar Pradesh"
    )

    if package3:
        # Location 1: Varanasi-Allahabad segment
        create_location(
            package_id=package3["id"],
            segment_name="Varanasi-Allahabad (KM 0-120)",
            chainage_start=0.0,
            chainage_end=120.0,
            start_lat=25.3176,  # Varanasi
            start_lng=82.9739,
            end_lat=25.4358,  # Allahabad
            end_lng=81.8463,
        )

        # Location 2: Allahabad-Kanpur segment
        create_location(
            package_id=package3["id"],
            segment_name="Allahabad-Kanpur (KM 120-320)",
            chainage_start=120.0,
            chainage_end=320.0,
            start_lat=25.4358,  # Allahabad
            start_lng=81.8463,
            end_lat=26.4499,  # Kanpur
            end_lng=80.3319,
        )


def seed_mumbai_pune_expressway():
    """Seed Mumbai-Pune Expressway project"""
    print("\n" + "=" * 60)
    print("Seeding: Mumbai-Pune Expressway")
    print("=" * 60)

    # Create project
    project = create_project(
        name="Mumbai-Pune Expressway Maintenance",
        state="Maharashtra",
        corridor_name="Mumbai-Pune Expressway (NH-48)",
        start_lat=18.9220,  # Mumbai
        start_lng=72.8347,
        end_lat=18.5204,  # Pune
        end_lng=73.8567,
    )

    if not project:
        return

    # Package 1: Mumbai-Lonavala Section
    package1 = create_package(
        project_id=project["id"],
        name="Mumbai-Lonavala Section",
        region="Mumbai Metropolitan Region",
    )

    if package1:
        # Location 1: Kalyan-Khopoli segment
        create_location(
            package_id=package1["id"],
            segment_name="Kalyan-Khopoli (KM 0-40)",
            chainage_start=0.0,
            chainage_end=40.0,
            start_lat=19.2403,  # Kalyan
            start_lng=73.1305,
            end_lat=18.7854,  # Khopoli
            end_lng=73.3457,
        )

        # Location 2: Khopoli-Lonavala segment
        create_location(
            package_id=package1["id"],
            segment_name="Khopoli-Lonavala (KM 40-70)",
            chainage_start=40.0,
            chainage_end=70.0,
            start_lat=18.7854,  # Khopoli
            start_lng=73.3457,
            end_lat=18.7537,  # Lonavala
            end_lng=73.4087,
        )

    # Package 2: Lonavala-Pune Section
    package2 = create_package(
        project_id=project["id"], name="Lonavala-Pune Section", region="Pune District"
    )

    if package2:
        # Location 1: Lonavala-Talegaon segment
        create_location(
            package_id=package2["id"],
            segment_name="Lonavala-Talegaon (KM 0-25)",
            chainage_start=0.0,
            chainage_end=25.0,
            start_lat=18.7537,  # Lonavala
            start_lng=73.4087,
            end_lat=18.7351,  # Talegaon
            end_lng=73.6765,
        )

        # Location 2: Talegaon-Pune segment
        create_location(
            package_id=package2["id"],
            segment_name="Talegaon-Pune (KM 25-55)",
            chainage_start=25.0,
            chainage_end=55.0,
            start_lat=18.7351,  # Talegaon
            start_lng=73.6765,
            end_lat=18.5204,  # Pune
            end_lng=73.8567,
        )


def seed_delhi_chandigarh_highway():
    """Seed Delhi-Chandigarh Highway project"""
    print("\n" + "=" * 60)
    print("Seeding: Delhi-Chandigarh Highway (NH-44)")
    print("=" * 60)

    # Create project
    project = create_project(
        name="Delhi-Chandigarh Highway Upgrade",
        state="Delhi, Haryana, Punjab",
        corridor_name="National Highway 44 (NH-44)",
        start_lat=28.7041,  # Delhi
        start_lng=77.1025,
        end_lat=30.7333,  # Chandigarh
        end_lng=76.7794,
    )

    if not project:
        return

    # Package 1: Delhi-Panipat Section
    package1 = create_package(
        project_id=project["id"],
        name="Delhi-Panipat Section",
        region="Delhi-Haryana Border",
    )

    if package1:
        # Location 1: Delhi-Sonipat segment
        create_location(
            package_id=package1["id"],
            segment_name="Delhi-Sonipat (KM 0-45)",
            chainage_start=0.0,
            chainage_end=45.0,
            start_lat=28.7041,  # Delhi
            start_lng=77.1025,
            end_lat=28.9931,  # Sonipat
            end_lng=77.0151,
        )

        # Location 2: Sonipat-Panipat segment
        create_location(
            package_id=package1["id"],
            segment_name="Sonipat-Panipat (KM 45-90)",
            chainage_start=45.0,
            chainage_end=90.0,
            start_lat=28.9931,  # Sonipat
            start_lng=77.0151,
            end_lat=29.3909,  # Panipat
            end_lng=76.9635,
        )

    # Package 2: Panipat-Chandigarh Section
    package2 = create_package(
        project_id=project["id"],
        name="Panipat-Chandigarh Section",
        region="Haryana-Punjab Border",
    )

    if package2:
        # Location 1: Panipat-Karnal segment
        create_location(
            package_id=package2["id"],
            segment_name="Panipat-Karnal (KM 0-60)",
            chainage_start=0.0,
            chainage_end=60.0,
            start_lat=29.3909,  # Panipat
            start_lng=76.9635,
            end_lat=29.6857,  # Karnal
            end_lng=76.9905,
        )

        # Location 2: Karnal-Ambala segment
        create_location(
            package_id=package2["id"],
            segment_name="Karnal-Ambala (KM 60-130)",
            chainage_start=60.0,
            chainage_end=130.0,
            start_lat=29.6857,  # Karnal
            start_lng=76.9905,
            end_lat=30.3782,  # Ambala
            end_lng=76.7821,
        )

        # Location 3: Ambala-Chandigarh segment
        create_location(
            package_id=package2["id"],
            segment_name="Ambala-Chandigarh (KM 130-180)",
            chainage_start=130.0,
            chainage_end=180.0,
            start_lat=30.3782,  # Ambala
            start_lng=76.7821,
            end_lat=30.7333,  # Chandigarh
            end_lng=76.7794,
        )


def main():
    """Run all seed functions"""
    print("\n" + "=" * 60)
    print("     VisionRoad Database Seeding Script")
    print("=" * 60)
    print(f"Target API: {BASE_URL}")
    print("=" * 60)

    try:
        # Test connection
        response = requests.get("http://localhost:8000/")
        print("✓ API server is running\n")
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to API server!")
        print("  Make sure the server is running: uvicorn main:app --reload")
        return

    # Seed all projects
    seed_kolkata_jaipur_highway()
    seed_mumbai_pune_expressway()
    seed_delhi_chandigarh_highway()

    print("\n" + "=" * 60)
    print("✓ Seeding complete!")
    print("=" * 60)
    print("\nTest the APIs:")
    print(f"  - List projects:  GET {BASE_URL}/projects")
    print(f"  - List packages:  GET {BASE_URL}/packages")
    print(f"  - List locations: GET {BASE_URL}/locations")
    print(f"  - Project summary: GET {BASE_URL}/summary/projects/<project_id>")
    print(f"  - API docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
