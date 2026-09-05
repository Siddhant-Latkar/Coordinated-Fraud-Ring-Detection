def evaluate_rules(transaction: dict, stats: dict) -> tuple[float, list[str]]:
    """
    Human-readable, deterministic rule layer that sits alongside the ML
    model. Field names here match src/database.get_entity_stats() and
    src/features.py exactly.
    """
    evidence = []
    points = 0.0

    amount = transaction["amount"]
    device_users_24h = stats["device_unique_users_24h"]
    ip_users_24h = stats["ip_unique_users_24h"]
    device_users_30d = stats["device_unique_users_30d"]
    ip_users_30d = stats["ip_unique_users_30d"]
    device_velocity = stats["device_txn_count_24h"]
    ip_velocity = stats["ip_txn_count_24h"]

    if amount >= 1000:
        points += 0.20
        evidence.append(f"High-value transaction: \u20b9{amount:,.2f}")

    # Fast burst: same device/IP, many people, within a day.
    if device_users_24h >= 2:
        points += 0.30
        evidence.append(
            f"Device shared by {device_users_24h} other user(s) in the past 24 hours"
        )

    if ip_users_24h >= 2:
        points += 0.25
        evidence.append(
            f"IP address shared by {ip_users_24h} other user(s) in the past 24 hours"
        )

    # Slow drip: same device/IP resurfacing across many people over a
    # month -- this is the pattern that a 24h-only check misses.
    if device_users_30d >= 3:
        points += 0.30
        evidence.append(
            f"Device linked to {device_users_30d} other user(s) in the past 30 days"
        )

    if ip_users_30d >= 3:
        points += 0.25
        evidence.append(
            f"IP address linked to {ip_users_30d} other user(s) in the past 30 days"
        )

    if device_velocity >= 5:
        points += 0.20
        evidence.append(
            f"Device generated {device_velocity} prior transactions in 24 hours"
        )

    if ip_velocity >= 5:
        points += 0.20
        evidence.append(
            f"IP address generated {ip_velocity} prior transactions in 24 hours"
        )

    rule_score = min(points, 1.0)

    if not evidence:
        evidence.append("No high-confidence rule signal detected")

    return rule_score, evidence
