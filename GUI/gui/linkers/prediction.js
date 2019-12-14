let {PythonShell} = require('python-shell')
var path = require("path")


function get_weather() {

  var time = document.getElementById("time").value
  var date = document.getElementById("date").value
  // name = city
  var options = {
    scriptPath : path.join(__dirname, '/../engine/'),
    args : [time,date]
  }

  let pyshell = new PythonShell('prediction.py', options);


  pyshell.on('message', function(message) {
     console.log(message.length);
     
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
