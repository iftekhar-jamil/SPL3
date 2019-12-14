let {PythonShell} = require('python-shell')
var path = require("path")


function get_weather() {

 
  var date = document.getElementById("date").value
 
  var options = {
    scriptPath : path.join(__dirname, '/../engine/'),
    args : [date]
  }

  let pyshell = new PythonShell('analyze.py', options);


  pyshell.on('message', function(message) {
      console.log(message);
     let arr = message.substring(1, message.length - 1).split(",");
     console.log(arr);
     var element = document.getElementById("im");
     if(element!=null)
     element.parentNode.removeChild(element);
     
     if(document.getElementById("line-chart")==undefined){
      var elem = document.createElement('canvas');
      elem.setAttribute("height", "400px");
      elem.setAttribute("width", "500px");
      elem.setAttribute("id","line-chart");
      document.getElementById("find").appendChild(elem);
     }
  
     new Chart(document.getElementById("line-chart"), {
      type: 'line',
      data: {
        labels: ['00:00','02:00','04:00','06:00','08:00','10:00','12:00','14:00','16:00','18:00','20,00','22.00'],
        datasets: [{ 
            data: arr,
            label: "Traffic Intensity",
            backgroundColor: '#c46998',
            borderColor: "#00000",
            fill: true
          }, 
        ]
      },
      options: {
          title: {
            display: true,
            text: 'Traffic Intensity per hour'
          },
          scales: {
            yAxes: [{
                display: true,
                ticks: {
                    suggestedMax: 33,
                    beginAtZero: true 
                }
            }]
        }
      }
    }); 

















     
  })
  
  document.getElementById("date").value = "";
  
}
