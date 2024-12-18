using LinearAlgebra, RowEchelon


function HITS_rank(websites_list::Vector{Tuple{String, Int64}}, website_links, depth::Int64=-1)
 num_websites = 0
 for (website, index) in websites_list
   num_websites += 1
 end


 if num_websites == 0
   error("Bad website count!")
 end


 #Once we get num_websites, we know the dimensions for A


 A = zeros(Float64, (num_websites, num_websites))


 for (website, index) in websites_list
   link_count = 0
   if get(website_links, (website, index), -1) != -1
     w_l = get(website_links, (website, index), -1)
     #set values inside A for column col
     for (website_link, website_link_index) in w_l
       A[website_link_index + num_websites * (index - 1)] = 1.0
     end
   end
 end


 S_1 = A' * A
 S_2 = A * A'


 #Initial value of v chosen by convention. v represents the hub scores of the webpages
 v = normalize([1.0 for r in 1:num_websites])
 # u represents the authority scores of the webpages
 u = normalize(A * v)


 if depth != -1
   n = depth
   while (n != 0)
     v = normalize(S_1 * v)
     u = normalize(S_2 * u)
     n -= 1;
   end 
 else
   #if we care about authority
   v = svd(S_1).U[:, 1]
   u = svd(S_2).U[:, 1]
   for i in 1:num_websites
     v[i] = abs(v[i])
     u[i] = abs(u[i])
   end
 end


 # Returning u because in this case more interested in the hub weight of the webpages,                                #in my case
 return u
end


function pagerank_func(websites_list::Vector{Tuple{String, Int64}}, website_links, depth::Int64=-1)
 num_websites = 0
 for (website, index) in websites_list
   num_websites += 1
 end


 if num_websites == 0
   error("Bad website count!")
 end


 #Once we get num_websites, we know the dimensions for A


 A = zeros(Float64, (num_websites, num_websites))
  #Now need to rewrite the values of A based on how many websites each site is #pointing to
 for (website, index) in websites_list
   link_count = 0
   if get(website_links, (website, index), -1) != -1
     w_l = get(website_links, (website, index), -1)
     for (website_link, website_link_index) in w_l
       link_count += 1
     end
     #Because in if statement, link_count will never be 0 to cause div by 0 error
     weight = 1 / link_count
     #set values inside A for column col
     for (website_link, website_link_index) in w_l
       A[website_link_index + num_websites * (index - 1)] = weight
     end
   else
     for col_index in 1:num_websites
       A[col_index + num_websites * (index - 1)] = 1 / num_websites
     end
   end
 end
 #damping factor
 p = 0.15
 B = [1 / num_websites for r in 1:num_websites, c in 1:num_websites]


 #Pagerank Matrix
 M = (1 - p) * A + p * B
  if (depth != -1)
   #we can then define vector v
   v = [1/num_websites for r in 1:num_websites]
   return (M^depth)*v
 else
   # This is a Markov Matrix, so we know that there is exactly 1 eigenvalue of 1.
   # Now to find its eigenvector
  
   I = [(r == c)+0.0 for r in 1:num_websites, c in 1:num_websites]
   C = rref_with_pivots(M - I)
   k = 1
   while (k != num_websites && C[1][:, k] == C[1][:, C[2][k]])
     k += 1;
   end


   N = [1.0 for r in 1:num_websites]
   for i in 1:(num_websites - 1)
     N[i] = -C[1][:, k][i]
   end


   #Now to get N to the correct size, for consistency
   N_sum = 0
   for i in 1:num_websites
      N_sum += N[i]
   end


   return N / N_sum
 end
end


w_list = [("sadness.com", 1), ("depression.com", 2), ("midterms.com", 3)]


w_links = Dict([(("sadness.com", 1), [("depression.com", 2), ("midterms.com", 3)]),
               (("depression.com", 2),[("midterms.com", 3)])])


print(pagerank_func(w_list, w_links))
print(HITS_rank(w_list, w_links))
