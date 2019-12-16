let {PythonShell} = require('python-shell')
var path = require("path")
var loadingSpinner = require('loading-spinner')



function get_weather() {

  var time = document.getElementById("time").value
  var date = document.getElementById("date").value
  var holiday = document.getElementById("holiday").checked
  
  var predPath = path.join(__dirname, '../engine/prediction.py')

  // name = city
  var options = {
   // scriptPath : path.join(__dirname, '/../engine/'),
    args : [time,date,holiday]
  }

  

  let pyshell = new PythonShell(predPath, options);

  loadingSpinner.start(
   
  );
  

  pyshell.on('message', function(message) {
      console.log(message);
     
     if(document.getElementById("output")!=undefined){
      var elem1 = document.getElementById("output");
      elem1.parentNode.removeChild(elem1);
     }
        var elem = document.createElement('img');
        elem.setAttribute("src", "data:image/jpeg;base64, "+message);
        elem.setAttribute("height", "400px");
        elem.setAttribute("width", "600px");
        elem.setAttribute("id","new");
     
    // // newImage.src = message;
      document.getElementById("new").src = "data:image/jpeg;base64, "+message;
      // document.getElementById("New").innerHTML = elem;
  })
  document.getElementById("time").value = "";
  document.getElementById("date").value = "";
  
}
