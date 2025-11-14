Potential plan to deal with evaluation:
_source.suricata.eve.event_type == "alert" or
_source.suricata.eve.flow.alerted == true
I think doing this and then creating a visual to show reconstruction error and color the data points differently based on whether they're suspicious or not would be a good visual.
Ex: plot reconstruction error, datapoints that are blue are likely not suspicious and red are suspicious (based on event type) then plot our threshold to see which red points we miss with our threshold

How to 'ensure' that training data is on benign data:
Probably want to start with a big dataset and then cull it down by filtering out "suspicious" things.
Ideas for how to filter the data out:
    * build a pipeline to feed ips into a public reputation website like virustotal or abuseipdb and remove any that are blocked or are reported X% and above confidence of malice
    * remove any connections over weird/suspicious port numbers that are likely to be attacked
    * high number of destinations in a short time (signs of scanning)
    * high number of short connections (signs of brute-force)
    * repeated failed connections with 0 bytes sent (signs of probing)
    * if we have geo location then anything that's international
    * definitely keep anything that passes on allowlisting api's (known good)
This gives us a dataset full of likely benign data. It can never be 100% confirmed benign, but as long as we have a good way to filter out the data we should have a solid dataset.
