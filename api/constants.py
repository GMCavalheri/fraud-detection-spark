"""
Reference data mirrored from spark_jobs/common.py. Duplicated (not imported)
on purpose: the API and Spark jobs ship as separate Docker images with no
shared filesystem, so this keeps the API container lightweight and
independently deployable at the cost of keeping these two lists in sync by
hand if the taxonomy ever changes.
"""

CATEGORIES = [
    "Electronics", "Groceries", "Restaurants", "Travel", "Fuel", "Fashion",
    "Entertainment", "Health & Pharmacy", "Home & Garden", "Online Services",
    "Utilities", "Jewelry",
]

CHANNELS = ["online", "pos", "atm"]

CITY_COORDS = {
    "New York": (40.7128, -74.0060), "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298), "London": (51.5074, -0.1278),
    "Manchester": (53.4808, -2.2426), "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050), "Madrid": (40.4168, -3.7038),
    "Sao Paulo": (-23.5505, -46.6333), "Rio de Janeiro": (-22.9068, -43.1729),
    "Brasilia": (-15.7939, -47.8828), "Tokyo": (35.6762, 139.6503),
    "Osaka": (34.6937, 135.5023), "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631), "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207), "Dubai": (25.2048, 55.2708),
    "Singapore": (1.3521, 103.8198), "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025), "Mexico City": (19.4326, -99.1332),
    "Lagos": (6.5244, 3.3792), "Johannesburg": (-26.2041, 28.0473),
    "Rome": (41.9028, 12.4964), "Amsterdam": (52.3676, 4.9041),
    "Lisbon": (38.7223, -9.1393), "Seoul": (37.5665, 126.9780),
    "Buenos Aires": (-34.6037, -58.3816), "Warsaw": (52.2297, 21.0122),
}
