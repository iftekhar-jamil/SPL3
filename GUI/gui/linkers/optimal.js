let {PythonShell} = require('python-shell')
var path = require("path")


function get_weather() {
  

  var date = document.getElementById("date").value
  var time1 = document.getElementById("time1").value
  var time2 = document.getElementById("time2").value
  var holiday = document.getElementById("holiday").checked
  // name = city
  var options = {
    scriptPath : path.join(__dirname, '../engine/'),
    args : [date,time1,time2, holiday]
  }

  let pyshell = new PythonShell('optimal.py', options);

  // debugger;
  pyshell.on('message', function(message) {
    //  console.log(message)
     var time = message.split(" ")[0];
     var img = message.split(" ")[1]; 
     time = Number(time);
     var decimal = time - Math.floor(time);
     time = Math.floor(time)+(decimal*60.0)/100.0;
    //  console.log(message.slice(0,10)); 
     var h = document.createElement("H1"); 
     time = time.toFixed(2);
     time = time.toString();
     if(time.length==4)
        time = "0"+time;
      // console.log("asas")  
     var t = document.createTextNode("Most optimal time is "+time); 
     h.appendChild(t);
     var element = document.getElementById("pic");
     if(element!=undefined)
      element.parentNode.removeChild(element);
     document.getElementById("output").appendChild(h);
     var elem = document.createElement('img');
        elem.setAttribute("src", "data:image/jpeg;base64, "+img);
        elem.setAttribute("height", "400px");
        elem.setAttribute("width", "500px");
        elem.setAttribute("id","output");
        document.getElementById("output").appendChild(elem);

  })
  
  document.getElementById("date").value = "";
  document.getElementById("time1").value = "";
  document.getElementById("time2").value = "";
  
}
