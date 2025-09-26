from backend.app.services.engine import BeerLine


def test_inventory_position_matches_onhand_minus_backlog():
    line = BeerLine()

    # Seed different on-hand/backlog values to ensure non-trivial inventory positions.
    for node in line.nodes:
        node.inventory = 5
        node.backlog = 1

    retailer = line.nodes[0]
    retailer.inventory = 3
    retailer.backlog = 4

    stats = line.tick(customer_demand=5)

    for node in line.nodes:
        role_stats = stats[node.name]
        assert role_stats["inventory_position"] == role_stats["inventory_after"] - role_stats["backlog_after"]
        assert role_stats["inventory_position"] == node.inventory - node.backlog
        assert role_stats["inventory_position_with_pipeline"] == node.inventory_position
