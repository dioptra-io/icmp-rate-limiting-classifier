# Limited Ltd.

Limited Ltd. is a new tool for performing alias resolution, the task to group IP addresses into routers.
It is based on ICMP rate limiting.



## Getting Started

The software is composed of two components, the prober (written in C++) and the classifier written in Python 3.

To get the prober:
```
git clone https://gitlab.planet-lab.eu/cartography/icmp-rate-limiting.git
```

To get the classifier:

```
git clone https://gitlab.planet-lab.eu/cartography/icmp-rate-limiting-classifier.git
```

### Prerequisites

MMLPT is written in Python, and uses different libraries to work.

To craft, send and parse packets, it uses scapy:
```
sudo pip install scapy
```

If you are on OSX, scapy does not work natively and you must install libdnet and its python wrapper. It works for python2+.

For python3+, libdnet has no python wrapper at the moment, so we recommend you using kamene, which is a fork of scapy supporting
python3+.

We will remove the dependency to kamene as soon as libdnet has a wrapper for python3 working with scapy.
```
sudo pip install kamene
```


To provide a rich graph view, and to compute the IP and router level graphs, MMLPT uses graph-tool, which is a boost C++ library with a python wrapper.

Follow the instructions in the following link to install graph-tool. As the C++ Library makes an extensive use of metaprogramming and templates, it can take a while (up to 30 minutes on a recent laptop) to compile graph-tool:

[Graph-tool](https://graph-tool.skewed.de)

### Installing

To install then, just tap:

```
sudo python setup.py install
```

## Running
To run a simple MDA-Lite traceroute at IP level, tap
```
sudo python MDALite.py destination
```
To run MDA-Lite traceroute at both IP and router level, tap:
```
sudo python MDALite.py -a destination
```
You can tap the following command for the different options

```
sudo python MDALite.py --help
```

Recall that on very large topologies, the measurement can take few minutes, up to 10-15 on very complex topologies.

## Exploring traceroute results
You should have this kind of output:

(0) : 127.0.0.1 -> ['192.168.0.254']

(1) : 192.168.0.254 -> ['137.194.164.254']

(2) : 137.194.164.254 -> ['137.194.4.240']

(3) : 137.194.4.240 -> ['212.73.200.45']

(4) : 212.73.200.45 -> ['4.69.161.114', '4.69.161.110']

(5) : 4.69.161.114 -> ['4.68.111.194']

4.69.161.110 -> ['4.68.111.194']

(6) : 4.68.111.194 -> ['62.115.122.10', '62.115.122.4']

(7) : 62.115.122.10 -> ['62.115.122.139']

62.115.122.4 -> ['62.115.123.12']

(8) : 62.115.123.12 -> ['62.115.138.236']

62.115.122.139 -> ['62.115.138.104']

(9) : 62.115.138.104 -> ['62.115.133.31', '62.115.136.37', '62.115.141.205', '62.115.119.115', '62.115.136.113', '62.115.139.107', '62.115.114.171', '62.115.136.35']

62.115.138.236 -> ['62.115.136.107', '62.115.133.29', '62.115.136.21', '62.115.136.23', '62.115.114.167', '62.115.141.201', '62.115.119.113', '62.115.136.25']

(10) : 62.115.136.107 -> ['213.248.79.106']

62.115.133.31 -> ['213.248.79.106']

62.115.133.29 -> ['213.248.79.106']

62.115.136.37 -> ['213.248.79.106']

62.115.141.205 -> ['213.248.79.106']

62.115.136.21 -> ['213.248.79.106']

62.115.119.115 -> ['213.248.79.106']

62.115.136.23 -> ['213.248.79.106']

62.115.136.113 -> ['213.248.79.106']

62.115.139.107 -> ['213.248.79.106']

62.115.114.171 -> ['213.248.79.106']

62.115.114.167 -> ['213.248.79.106']

62.115.141.201 -> ['213.248.79.106']

62.115.119.113 -> ['213.248.79.106']

62.115.136.25 -> ['213.248.79.106']

62.115.136.35 -> ['213.248.79.106']

(11) : 213.248.79.106 -> ['195.12.233.115']

(12) : 195.12.233.115

Routers found :

['62.115.136.113', '62.115.136.107']

['62.115.136.23', '62.115.133.31', '62.115.141.201', '62.115.136.21', '62.115.136.25']

['62.115.133.29', '62.115.119.113', '62.115.119.115']

['62.115.136.35', '62.115.136.37']

If the text output is not verbose enough, you can ask for a serialized version of the graph that basically keeps all the informations of the traceroute with the -o option

```
sudo python MDALite.py -o traceroute.xml -a destination
```

```python
from graph_tool import *
g = load_graph("/path/to/serialized/graphs.xml")

# Manipulate the graph via its properties.
v_properties = g.vertex_properties
e_properties = g.edge_properties
g_properties = g.graph_properties

# Example: ip_ids, print ip_ids of an ip_address
ip_address = g.vertex_properties["ip_address"]
ip_ids = g.vertex_properties["ip_ids"]
for v in g.vertices():
    if ip_address[v] == "an_ip_of_your_choice_in_the_traceroute":
        print ip_ids[v]

```

## Deploying in a docker container.
Dockerfile provides an example of a Docker script that allows you to deploy
MMLPT in a Dockercontainer (in the example with an ubuntu image).
To build the container, launch the command:
```
docker build --build-arg "CACHE_DATE=$(date)" -t "name/of/your/docker/image" .
```
Then push it to docker repository:
```
docker push name/of/your/docker/image
```

To run it:
```
docker run -d name/of/your/docker/image
```

<!--- ## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
-->
