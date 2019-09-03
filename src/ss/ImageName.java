package ss;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class ImageName {
	
	
	public String getDateAndTime()
	{
		 Date date = new Date();
		 DateFormat dateFormat = new SimpleDateFormat("dd_MM_yyyy HH_mm_ss_a");
	
		return dateFormat.format(date);
	}
	
	public String getTime()
	{
		Date date = new Date();
		DateFormat dateFormat=new SimpleDateFormat("hh:mm:ss");
		return dateFormat.format(date);
	}
	
	public String getImageName()
	{
		return getDateAndTime();
	}
	
}
