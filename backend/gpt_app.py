from flask import Flask, request, jsonify
from llm_agent.daybreak_tbg_agent import call_beer_game_gpt

app = Flask(__name__)

@app.post("/order")
def get_order():
    data = request.get_json(force=True)
    user_message = (
        f"{data['role']}, Turn {data['turn']}, "
        f"On-hand: {data['on_hand']}, Backlog: {data['backlog']}, "
        f"Expected deliveries: {data['expected_deliveries']}, Demand: {data['demand']}"
    )
    order, parsed = call_beer_game_gpt(user_message)
    comment = parsed.get("comment") if parsed else None
    return jsonify({"order": order, "comment": comment, "data": parsed})

if __name__ == "__main__":
    app.run(debug=True)
