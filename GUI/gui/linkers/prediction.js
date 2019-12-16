let {PythonShell} = require('python-shell')
var path = require("path")




function get_weather() {

  document.getElementById("new").src = "https://icon-library.net/images/loading-icon-animated-gif/loading-icon-animated-gif-19.jpg"
        var time = document.getElementById("time").value
        var date = document.getElementById("date").value
        var holiday = document.getElementById("holiday").checked
        
        var predPath = path.join(__dirname, '../engine/prediction.py')

        // name = city
        var options = {
        // scriptPath : path.join(__dirname, '/../engine/'),
          args : [time,date,holiday]
        }

        
        // document.getElementById("new").setAttribute("display", "none");
        // document.getElementById("loader").setAttribute("display", "block");
         
        let pyshell = new PythonShell(predPath, options);
                                 
        pyshell.on('message', function(message) {
            console.log(message);
            document.getElementById("loader").setAttribute("display", "none");
            if(document.getElementById("output")!=undefined){
              var elem1 = document.getElementById("output");
              elem1.parentNode.removeChild(elem1);
            }
                var elem = document.createElement('img');
                elem.setAttribute("src", "data:image/jpeg;base64, "+message);
                elem.setAttribute("height", "400px");
                elem.setAttribute("width", "600px");
                elem.setAttribute("id","new");
                document.getElementById("new").setAttribute("display", "block");
            
            // // newImage.src = message;
               document.getElementById("new").src = "data:image/jpeg;base64, "+message;
            // document.getElementById("New").innerHTML = elem;
        })
          document.getElementById("time").value = "";
          document.getElementById("date").value = "";
  
}
