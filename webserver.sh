python2 app.py "homework.json" "results.csv"
echo > test.html
cat results.csv
echo "<html><body><h1>Results:+"echo results.csv+"</h1></body></html>" > test.html
python -m http.server 8080
open http://0.0.0.0:8080/test.html
