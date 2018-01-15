#include "parser.h"
#include "vw.h"
#include "parse_regressor.h"

#include <sys/time.h>

using namespace std;

uint64_t get_time_us() {
  struct timeval tv; 
  gettimeofday(&tv, NULL);
  return 1000000UL * tv.tv_sec + tv.tv_usec;
}


void dispatch_example(vw& all, example& ec)
{
  all.learn(&ec);
  all.l->finish_example(all, ec);
}

namespace prediction_type
{
#define CASE(type) case type: return #type;

const char* to_string(prediction_type_t prediction_type)
{
  switch (prediction_type)
  {
    CASE(scalar)
    CASE(scalars)
    CASE(action_scores)
    CASE(action_probs)
    CASE(multiclass)
    CASE(multilabels)
    CASE(prob)
    CASE(multiclassprobs)
  default: return "<unsupported>";
  }
}
}

namespace LEARNER
{
void process_example(vw& all, example* ec)
{
  if (ec->indices.size() > 1) // 1+ nonconstant feature. (most common case first)
    dispatch_example(all, *ec);
  else if (ec->end_pass)
  {
    all.l->end_pass();
    VW::finish_example(all, ec);
  }
  else if (ec->tag.size() >= 4 && !strncmp((const char*) ec->tag.begin(), "save", 4))
  {
    // save state command

    string final_regressor_name = all.final_regressor_name;

    if ((ec->tag).size() >= 6 && (ec->tag)[4] == '_')
      final_regressor_name = string(ec->tag.begin()+5, (ec->tag).size()-5);

    if (!all.quiet)
      all.trace_message << "saving regressor to " << final_regressor_name << endl;
    save_predictor(all, final_regressor_name, 0);

    VW::finish_example(all,ec);
  }
  else // empty example
    dispatch_example(all, *ec);
}

template <class T, void(*f)(T, example*)> void generic_driver(vw& all, T context)
{
  example* ec = nullptr;

  unsigned long int count = 0;
  auto start = get_time_us();
  while ( all.early_terminate == false ) {
    if ((ec = VW::get_example(all.p)) != nullptr) {
      count++;
      if (count % 10000 == 0) {
        auto elapsed = get_time_us() - start;
        std::cout
          << std::fixed
          << "samples/sec: " << 1.0 * count / elapsed * 1000 * 1000
          << "count: " << count
          << "\n";
      }
      f(context, ec);
    }
    else
      break;
  }
  if (all.early_terminate) //drain any extra examples from parser.
    while ((ec = VW::get_example(all.p)) != nullptr)
      VW::finish_example(all, ec);
  all.l->end_examples();
}

void process_multiple(vector<vw*> alls, example* ec)
{
  std::cout << "process_multiple" << std::endl;
  // start with last as the first instance will free the example as it is the owner
  for (auto it = alls.rbegin(); it != alls.rend(); ++it)
    process_example(**it, ec);
}

void generic_driver(vector<vw*> alls)
{
  std::cout << "generic_driver1" << std::endl;
  generic_driver<vector<vw*>, process_multiple>(**alls.begin(), alls);

  // skip first as it already called end_examples()
  auto it = alls.begin();
  for (it++; it != alls.end(); it++)
    (*it)->l->end_examples();
}

void generic_driver(vw& all)
{ 
  std::cout << "generic_driver2" << std::endl;
  generic_driver<vw&, process_example>(all, all); }
}
