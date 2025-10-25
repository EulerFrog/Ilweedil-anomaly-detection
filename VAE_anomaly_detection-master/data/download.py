from opensearchpy import OpenSearch

host = "140.160.1.66"
port = 50001

# Create the client with SSL/TLS enabled, but hostname verification disabled.
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = False,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
)

breakpoint()

#query = {
#  'query': {
#        
#    }
#}



response = client.search(
    body = '*',
    index = '*'
)

breakpoint()
print(result)

