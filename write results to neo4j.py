import numpy as np

predicted_nodes = ["a", "b"]
predicted_scores = np.array([[[0.1, 0.8, 0.1], [0.4, 0.35, 0.25]]])  # a  # b
predicted_class = np.array(["Y", "X"])

predictions = [
    {"name": name, "scores": list(scores), "class": class_}
    for name, scores, class_ in zip(predicted_nodes, predicted_scores[0], predicted_class)
]
predictions

neo4j_graph.evaluate(
    """
    UNWIND $predictions AS prediction
    MATCH (n { name: prediction.name })
    SET n.predicted_class_scores = prediction.scores
    SET n.predicted_class = prediction.class
    """,
    {"predictions": predictions},
)

verification_data = neo4j_graph.run(
    "MATCH (n) RETURN n.name, n.predicted_class_scores, n.predicted_class"
).to_data_frame()

verification_data.sort_values("n.name")  # sort for ease of reference