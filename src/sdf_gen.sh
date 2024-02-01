#!/bin/bash

categories="vase trashbin couch bed drinkingutensil"

for category in $categories; do
    echo "Running script for category: $category"
    python3 create_sdf.py --category="$category" --mesh_type="simplified"
    wait $!
    echo "Script for category $category completed"
done

echo "All scripts completed"

