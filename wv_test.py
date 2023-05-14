import weaviate

client = weaviate.Client("http://localhost:8080")

is_live = client.is_ready()
print(is_live)
